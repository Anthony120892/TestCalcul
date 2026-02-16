import json
import os
import calendar
from datetime import date, timedelta
from io import BytesIO

import streamlit as st

# ============================================================
# CONFIG PAR DÉFAUT (fusion avec ris_rules.json si présent)
# => On UTILISE les TAUX ANNUELS comme référence (centime près)
# ============================================================
DEFAULT_ENGINE = {
    "version": "1.4",
    "config": {
        # Taux RIS ANNUELS (référence) ✅
        "ris_rates_annuel": {"cohab": 10513.60, "isole": 15770.41, "fam_charge": 21312.87},

        # (Optionnel) ancien champ laissé pour compat / info
        "ris_rates": {"cohab": 876.13, "isole": 1314.20, "fam_charge": 1776.07},

        # Immunisation simple (annuelle)
        "immunisation_simple_annuelle": {"cohab": 155.0, "isole": 250.0, "fam_charge": 310.0},

        # Art. 34 : taux "catégorie 1 à laisser" (mensuel)
        "art34": {"taux_a_laisser_mensuel": 876.13},

        # Prestations familiales (montant de référence indexable)
        "pf": {"pf_mensuel_defaut": 0.0},

        # Capitaux mobiliers (annuels)
        "capital_mobilier": {
            "t0_max": 6199.0,
            "t1_min": 6200.0,
            "t1_max": 12500.0,
            "t1_rate": 0.06,
            "t2_rate": 0.10
        },

        # Immobilier (RC non indexé) + coeff *3
        "immo": {
            "bati_base": 750.0,
            "bati_par_enfant": 125.0,
            "non_bati_base": 30.0,
            "coeff_rc": 3.0
        },

        # Exonérations socio-pro (mensuel / annuel)
        "socio_prof": {
            "max_mensuel": 309.48,
            "artistique_annuel": 3297.80,
        },

        # Cession de biens
        "cession": {
            "tranche_immunisee": 37200.0,
            "usufruit_ratio": 0.40,
            "abattements_annuels": {"cat1": 1250.0, "cat2": 2000.0, "cat3": 2500.0}
        },

        # ALE
        "ale": {"exon_mensuelle": 4.10}
    }
}


# ============================================================
# UTILITAIRES
# ============================================================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def r2(x: float) -> float:
    return float(round(float(x), 2))


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def normalize_engine(raw: dict) -> dict:
    raw = raw or {}
    engine = deep_merge(DEFAULT_ENGINE, raw)
    cfg = engine["config"]

    # Backward compat: exonerations -> immo
    if "exonerations" in cfg and "immo" in cfg:
        exo = cfg["exonerations"]
        cfg["immo"]["bati_base"] = float(exo.get("bati_base", cfg["immo"]["bati_base"]))
        cfg["immo"]["bati_par_enfant"] = float(exo.get("bati_par_enfant", cfg["immo"]["bati_par_enfant"]))
        cfg["immo"]["non_bati_base"] = float(exo.get("non_bati_base", cfg["immo"]["non_bati_base"]))

    # Assurer presence/typage
    if "ris_rates_annuel" not in cfg:
        cfg["ris_rates_annuel"] = {"cohab": None, "isole": None, "fam_charge": None}

    for k in ("cohab", "isole", "fam_charge"):
        cfg["ris_rates"][k] = float(cfg["ris_rates"].get(k, 0.0))
        cfg["immunisation_simple_annuelle"][k] = float(cfg["immunisation_simple_annuelle"].get(k, 0.0))
        if cfg["ris_rates_annuel"].get(k) is not None:
            cfg["ris_rates_annuel"][k] = float(cfg["ris_rates_annuel"][k])

    if "art34" not in cfg:
        cfg["art34"] = {}
    cfg["art34"]["taux_a_laisser_mensuel"] = float(
        cfg["art34"].get("taux_a_laisser_mensuel", cfg["ris_rates"].get("cohab", 0.0))
    )

    # PF indexables (valeur de référence)
    if "pf" not in cfg:
        cfg["pf"] = {"pf_mensuel_defaut": 0.0}
    cfg["pf"]["pf_mensuel_defaut"] = float(cfg["pf"].get("pf_mensuel_defaut", 0.0))

    return engine


def load_engine() -> dict:
    if os.path.exists("ris_rules.json"):
        with open("ris_rules.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        return normalize_engine(raw)
    return normalize_engine(DEFAULT_ENGINE)


def end_of_month(d: date) -> date:
    dim = calendar.monthrange(d.year, d.month)[1]
    return date(d.year, d.month, dim)


def next_day(d: date) -> date:
    return d + timedelta(days=1)


def date_in_same_month(d: date, ref: date) -> bool:
    return d.year == ref.year and d.month == ref.month


def safe_parse_date(x) -> date | None:
    if isinstance(x, date):
        return x
    if isinstance(x, str) and x.strip():
        try:
            return date.fromisoformat(x.strip())
        except Exception:
            return None
    return None


# ============================================================
# CAPITAUX MOBILIERS (annuel -> renvoie ANNUEL)
# ============================================================
def capital_mobilier_annuel(total_capital: float,
                            compte_commun: bool,
                            nb_titulaires: int,
                            categorie: str,
                            conjoint_compte_commun: bool,
                            part_fraction_custom: float,
                            cfg_cap: dict) -> float:
    total_capital = max(0.0, float(total_capital))

    if compte_commun:
        nb = max(1, int(nb_titulaires))
        numerator = 2 if (categorie == "fam_charge" and conjoint_compte_commun) else 1
        fraction = numerator / nb
    else:
        fraction = clamp01(part_fraction_custom)

    adj_total = total_capital * fraction

    t0_max = float(cfg_cap["t0_max"]) * fraction
    t1_min = float(cfg_cap["t1_min"]) * fraction
    t1_max = float(cfg_cap["t1_max"]) * fraction
    r1_ = float(cfg_cap["t1_rate"])
    r2_ = float(cfg_cap["t2_rate"])

    if adj_total <= t0_max:
        return 0.0

    annuel = 0.0
    tranche1 = max(0.0, min(adj_total, t1_max) - t1_min)
    annuel += tranche1 * r1_
    tranche2 = max(0.0, adj_total - t1_max)
    annuel += tranche2 * r2_
    return float(max(0.0, annuel))


# ============================================================
# IMMOBILIER (annuel)
# ============================================================
def immo_annuel_total(biens: list, enfants: int, cfg_immo: dict) -> float:
    biens_countes = [b for b in biens if not b.get("habitation_principale", False)]
    nb_bati = sum(1 for b in biens_countes if b.get("bati", True))
    nb_non_bati = sum(1 for b in biens_countes if not b.get("bati", True))

    exo_bati_total = float(cfg_immo["bati_base"]) + float(cfg_immo["bati_par_enfant"]) * max(0, int(enfants))
    exo_non_bati_total = float(cfg_immo["non_bati_base"])
    coeff = float(cfg_immo.get("coeff_rc", 3.0))

    total_annuel = 0.0

    for b in biens_countes:
        bati = bool(b.get("bati", True))
        rc = max(0.0, float(b.get("rc_non_indexe", 0.0)))
        frac = clamp01(b.get("fraction_droits", 1.0))
        rc_part = rc * frac

        if bati:
            exo_par_bien = (exo_bati_total * frac) / nb_bati if nb_bati > 0 else 0.0
        else:
            exo_par_bien = (exo_non_bati_total * frac) / nb_non_bati if nb_non_bati > 0 else 0.0

        base = max(0.0, (rc_part - exo_par_bien) * coeff)

        if b.get("hypotheque", False):
            interets = max(0.0, float(b.get("interets_annuels", 0.0))) * frac
            base -= min(interets, 0.5 * base)

        if b.get("viager", False):
            rente = max(0.0, float(b.get("rente_viagere_annuelle", 0.0))) * frac
            base -= min(rente, 0.5 * base)

        total_annuel += max(0.0, base)

    return float(max(0.0, total_annuel))


# ============================================================
# CESSION DE BIENS (annuel)
# ============================================================
def cession_biens_annuelle(cessions: list,
                           cas_particulier_tranche_37200: bool,
                           dettes_deductibles: float,
                           abatt_cat: str,
                           abatt_mois_prorata: int,
                           cfg_cession: dict,
                           cfg_cap: dict) -> float:
    total = 0.0
    for c in cessions:
        v = max(0.0, float(c.get("valeur_venale", 0.0)))
        if c.get("usufruit", False):
            v *= float(cfg_cession["usufruit_ratio"])
        total += v

    total = max(0.0, total - max(0.0, float(dettes_deductibles)))
    if cas_particulier_tranche_37200:
        total = max(0.0, total - float(cfg_cession["tranche_immunisee"]))

    abatt_annuel = float(cfg_cession["abattements_annuels"].get(abatt_cat, 0.0))
    mois = max(0, min(12, int(abatt_mois_prorata)))
    total = max(0.0, total - (abatt_annuel * (mois / 12.0)))

    t0_max = float(cfg_cap["t0_max"])
    t1_min = float(cfg_cap["t1_min"])
    t1_max = float(cfg_cap["t1_max"])
    r1_ = float(cfg_cap["t1_rate"])
    r2_ = float(cfg_cap["t2_rate"])

    if total <= t0_max:
        return 0.0

    annuel = 0.0
    tranche1 = max(0.0, min(total, t1_max) - t1_min)
    annuel += tranche1 * r1_
    tranche2 = max(0.0, total - t1_max)
    annuel += tranche2 * r2_
    return float(max(0.0, annuel))


# ============================================================
# REVENUS (exo socio-pro) - stock en ANNUEL
# ============================================================
def revenus_annuels_apres_exonerations(revenus_annuels: list, cfg_soc: dict) -> float:
    total_m = 0.0
    for r in revenus_annuels:
        a = max(0.0, float(r.get("montant_annuel", 0.0)))
        m = a / 12.0
        t = r.get("type", "standard")
        eligible = bool(r.get("eligible", True))

        if t in ("socio_prof", "etudiant") and eligible:
            ded = min(float(cfg_soc["max_mensuel"]), m)
            total_m += max(0.0, m - ded)
        elif t == "artistique_irregulier" and eligible:
            ded_m = float(cfg_soc["artistique_annuel"]) / 12.0
            total_m += max(0.0, m - min(ded_m, m))
        elif t == "ale":
            total_m += max(0.0, float(r.get("ale_part_excedentaire_mensuel", 0.0)))
        elif t == "prestations_familiales":
            total_m += m
        else:
            total_m += m
    return float(max(0.0, total_m * 12.0))


# ============================================================
# ART.34 — MODE SIMPLE (+ date de départ cohabitant)
# ============================================================
def normalize_art34_type(raw_type: str) -> str:
    t = (raw_type or "").strip().lower()
    aliases = {
        "debiteur direct 1": "debiteur_direct_1",
        "debiteur direct 2": "debiteur_direct_2",
        "debiteur_direct1": "debiteur_direct_1",
        "debiteur_direct2": "debiteur_direct_2",
        "partner": "partenaire",
    }
    return aliases.get(t, t)


def cohabitant_is_active_asof(c: dict, as_of: date) -> bool:
    dquit = safe_parse_date(c.get("date_quitte_menage"))
    if dquit is None:
        return True
    return as_of <= dquit


def cohabitants_art34_part_mensuelle_cpas(cohabitants: list,
                                         taux_a_laisser_mensuel: float,
                                         partage_active: bool,
                                         nb_demandeurs_a_partager: int,
                                         as_of: date) -> dict:
    """
    Règle CPAS attendue :
    - PARTENAIRE : ressources cohabitant comptées plein pot
    - DÉBITEURS (direct 1/2) : art.34 avec taux à laisser
    """
    taux = max(0.0, float(taux_a_laisser_mensuel))

    revenus_partenaire_m = 0.0
    nb_partenaire = 0

    revenus_debiteurs_m = 0.0
    nb_debiteurs = 0

    detail_partenaire = []
    detail_debiteurs = []

    for c in cohabitants:
        typ = normalize_art34_type(c.get("type", "autre"))
        if bool(c.get("exclure", False)):
            continue
        if not cohabitant_is_active_asof(c, as_of):
            continue

        revenu_ann = max(0.0, float(c.get("revenu_net_annuel", 0.0)))
        revenu_m = revenu_ann / 12.0

        if typ == "partenaire":
            revenus_partenaire_m += revenu_m
            nb_partenaire += 1
            detail_partenaire.append({"type": "partenaire", "mensuel": r2(revenu_m), "annuel": r2(revenu_ann)})
        elif typ in {"debiteur_direct_1", "debiteur_direct_2"}:
            revenus_debiteurs_m += revenu_m
            nb_debiteurs += 1
            detail_debiteurs.append({"type": typ, "mensuel": r2(revenu_m), "annuel": r2(revenu_ann)})
        else:
            pass

    part_debiteurs_m = max(0.0, revenus_debiteurs_m - (nb_debiteurs * taux))

    if partage_active:
        n = max(1, int(nb_demandeurs_a_partager))
        part_debiteurs_m_par_dem = part_debiteurs_m / n
    else:
        part_debiteurs_m_par_dem = part_debiteurs_m

    total_cohabitants_m = revenus_partenaire_m + part_debiteurs_m_par_dem

    return {
        "cohabitants_n_partenaire_pris_en_compte": int(nb_partenaire),
        "cohabitants_n_debiteurs_pris_en_compte": int(nb_debiteurs),
        "revenus_partenaire_mensuels_total": r2(revenus_partenaire_m),
        "revenus_debiteurs_mensuels_total": r2(revenus_debiteurs_m),
        "cohabitants_part_debiteurs_avant_partage_mensuel": r2(part_debiteurs_m),
        "cohabitants_part_debiteurs_apres_partage_mensuel": r2(part_debiteurs_m_par_dem),
        "cohabitants_part_a_compter_mensuel": r2(total_cohabitants_m),
        "cohabitants_part_a_compter_annuel": r2(total_cohabitants_m * 12.0),
        "detail_partenaire": detail_partenaire,
        "detail_debiteurs": detail_debiteurs,
        "taux_a_laisser_mensuel": r2(taux),
        "partage_active": bool(partage_active),
        "nb_demandeurs_partage": int(nb_demandeurs_a_partager),
    }


# ============================================================
# CALCUL GLOBAL — OFFICIEL CPAS (ANNUEL puis /12)
# + "as_of" pour gérer les cohabitants qui quittent le ménage
# ============================================================
def compute_officiel_cpas_annuel(answers: dict, engine: dict, as_of: date | None = None) -> dict:
    cfg = engine["config"]
    cat = answers.get("categorie", "isole")

    taux_ris_annuel = float(cfg.get("ris_rates_annuel", {}).get(cat) or 0.0)
    taux_ris_annuel = r2(taux_ris_annuel)
    taux_ris_m = r2(taux_ris_annuel / 12.0) if taux_ris_annuel > 0 else 0.0

    if as_of is None:
        as_of = answers.get("date_demande", date.today())
        if not isinstance(as_of, date):
            as_of = date.today()

    revenus_demandeur_annuels = revenus_annuels_apres_exonerations(
        answers.get("revenus_demandeur_annuels", []),
        cfg["socio_prof"]
    )
    if bool(answers.get("couple_demandeur", False)):
        revenus_conjoint_annuels = revenus_annuels_apres_exonerations(
            answers.get("revenus_conjoint_annuels", []),
            cfg["socio_prof"]
        )
        revenus_demandeur_annuels += revenus_conjoint_annuels
    revenus_demandeur_annuels = r2(revenus_demandeur_annuels)

    cap_ann = r2(capital_mobilier_annuel(
        total_capital=answers.get("capital_mobilier_total", 0.0),
        compte_commun=answers.get("capital_compte_commun", False),
        nb_titulaires=answers.get("capital_nb_titulaires", 1),
        categorie=cat,
        conjoint_compte_commun=answers.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=answers.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    ))

    immo_ann = r2(immo_annuel_total(
        biens=answers.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    ))

    ces_ann = r2(cession_biens_annuelle(
        cessions=answers.get("cessions", []),
        cas_particulier_tranche_37200=answers.get("cession_cas_particulier_37200", False),
        dettes_deductibles=answers.get("cession_dettes_deductibles", 0.0),
        abatt_cat=answers.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=answers.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    ))

    art34 = cohabitants_art34_part_mensuelle_cpas(
        cohabitants=answers.get("cohabitants_art34", []),
        taux_a_laisser_mensuel=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        partage_active=bool(answers.get("partage_enfants_jeunes_actif", False)),
        nb_demandeurs_a_partager=int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
        as_of=as_of
    )

    pf_m = r2(max(0.0, float(answers.get("prestations_familiales_a_compter_mensuel", 0.0))))
    pf_ann = r2(pf_m * 12.0)

    avantage_nature_m = r2(max(0.0, float(answers.get("avantage_nature_logement_mensuel", 0.0))))
    avantage_nature_ann = r2(avantage_nature_m * 12.0)

    total_avant_annuel = r2(
        revenus_demandeur_annuels
        + cap_ann
        + immo_ann
        + ces_ann
        + art34["cohabitants_part_a_compter_annuel"]
        + pf_ann
        + avantage_nature_ann
    )

    immu_ann = 0.0
    if taux_ris_annuel > 0 and total_avant_annuel < taux_ris_annuel:
        immu_ann = float(cfg["immunisation_simple_annuelle"].get(cat, 0.0))
    immu_ann = r2(immu_ann)

    total_apres_annuel = r2(max(0.0, total_avant_annuel - immu_ann))

    ris_annuel = r2(max(0.0, taux_ris_annuel - total_apres_annuel) if taux_ris_annuel > 0 else 0.0)
    ris_mensuel = r2(ris_annuel / 12.0)

    return {
        "mode_calcul": "OFFICIEL_CPAS_ANNUEL",
        "categorie": cat,
        "enfants_a_charge": int(answers.get("enfants_a_charge", 0)),
        "couple_demandeur": bool(answers.get("couple_demandeur", False)),

        "partage_enfants_jeunes_actif": bool(answers.get("partage_enfants_jeunes_actif", False)),
        "nb_enfants_jeunes_demandeurs": int(answers.get("nb_enfants_jeunes_demandeurs", 1)),

        "revenus_demandeur_annuels": float(revenus_demandeur_annuels),
        "capitaux_mobiliers_annuels": float(cap_ann),
        "immo_annuels": float(immo_ann),
        "cession_biens_annuelle": float(ces_ann),

        **art34,

        "prestations_familiales_a_compter_mensuel": float(pf_m),
        "prestations_familiales_a_compter_annuel": float(pf_ann),

        "avantage_nature_logement_mensuel": float(avantage_nature_m),
        "avantage_nature_logement_annuel": float(avantage_nature_ann),

        "total_ressources_avant_immunisation_simple_annuel": float(total_avant_annuel),
        "taux_ris_annuel": float(taux_ris_annuel),
        "immunisation_simple_annuelle": float(immu_ann),
        "total_ressources_apres_immunisation_simple_annuel": float(total_apres_annuel),
        "ris_theorique_annuel": float(ris_annuel),

        "taux_ris_mensuel_derive": float(taux_ris_m),
        "ris_theorique_mensuel": float(ris_mensuel),
        "as_of": str(as_of),
    }


# ============================================================
# SEGMENTS CPAS DU 1ER MOIS
# ============================================================
def compute_first_month_segments(answers: dict, engine: dict) -> dict:
    d_dem = answers.get("date_demande", date.today())
    if not isinstance(d_dem, date):
        d_dem = date.today()

    eom = end_of_month(d_dem)
    days_in_month = calendar.monthrange(d_dem.year, d_dem.month)[1]

    change_points = []
    for c in answers.get("cohabitants_art34", []) or []:
        dq = safe_parse_date(c.get("date_quitte_menage"))
        if dq is None:
            continue
        if date_in_same_month(dq, d_dem) and dq >= d_dem and dq < eom:
            change_points.append(next_day(dq))

    change_points = sorted(set(change_points))
    boundaries = [d_dem] + change_points + [next_day(eom)]

    segments = []
    total_first_month = 0.0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end_excl = boundaries[i + 1]
        end = end_excl - timedelta(days=1)
        if end < start:
            continue

        seg_days = (end - start).days + 1
        prorata = seg_days / days_in_month

        res_seg = compute_officiel_cpas_annuel(answers, engine, as_of=start)
        ris_m = float(res_seg["ris_theorique_mensuel"])
        amount = r2(ris_m * prorata)

        total_first_month = r2(total_first_month + amount)
        segments.append({
            "du": str(start),
            "au": str(end),
            "jours": int(seg_days),
            "prorata": float(prorata),
            "ris_mensuel": r2(ris_m),
            "montant_segment": float(amount),
            "as_of": str(start),
            "_detail_res": res_seg,   # 🔥 pour PDF ultra fidèle
        })

    ref_mois_suivants = boundaries[-2] if len(boundaries) >= 2 else d_dem
    res_suivants = compute_officiel_cpas_annuel(answers, engine, as_of=ref_mois_suivants)

    return {
        "date_demande": str(d_dem),
        "jours_dans_mois": int(days_in_month),
        "reference_mois_suivants": str(ref_mois_suivants),
        "ris_mois_suivants": float(res_suivants["ris_theorique_mensuel"]),
        "segments": segments,
        "ris_1er_mois_total": float(total_first_month),
        "detail_mois_suivants": res_suivants,
    }


# ============================================================
# PDF — VERSION "CPAS" (2 pages, ligne par ligne)
# ============================================================
def euro(x: float) -> str:
    x = float(x or 0.0)
    s = f"{x:,.2f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def date_fr(iso: str) -> str:
    try:
        y, m, d = iso.split("-")
        return f"{d}/{m}/{y}"
    except Exception:
        return str(iso)


def _safe(s) -> str:
    return (s or "").replace("\n", " ").strip()


def make_decision_pdf_cpas(
    dossier_label: str,
    answers_snapshot: dict,
    res_mois_suivants: dict,
    seg_first_month: dict | None = None,
    logo_path: str = "logo.png",
) -> BytesIO | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image, ListFlowable, ListItem, PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
    except Exception:
        return None

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.8*cm, rightMargin=1.8*cm,
        topMargin=1.4*cm, bottomMargin=1.4*cm
    )

    styles = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=styles["Normal"], fontName="Helvetica", fontSize=10, leading=13)
    small = ParagraphStyle("small", parent=base, fontSize=9, leading=12, textColor=colors.grey)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=18, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=14, spaceBefore=10, spaceAfter=4)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2)

    story = []

    # Header logo + titre
    logo_elem = None
    if logo_path and os.path.exists(logo_path):
        logo_elem = Image(logo_path, width=3.0*cm, height=3.0*cm)

    header_data = [
        [logo_elem if logo_elem else Paragraph("", base), Paragraph("Calcul du RI", h1)],
        ["", Paragraph(f"Dossier : <b>{_safe(dossier_label)}</b>", base)],
    ]
    header_tbl = Table(header_data, colWidths=[3.2*cm, 13.0*cm])
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 8))

    # Meta
    story.append(Paragraph(
        f"Catégorie : <b>{res_mois_suivants.get('categorie','')}</b> — "
        f"Taux RI annuel (référence) : <b>{euro(res_mois_suivants.get('taux_ris_annuel',0))} €</b>",
        base
    ))
    story.append(Paragraph(
        f"Taux RI mensuel (dérivé) : <b>{euro(res_mois_suivants.get('taux_ris_mensuel_derive',0))} €</b>",
        base
    ))
    story.append(Spacer(1, 10))

    def bullets(lines: list[str]):
        items = [ListItem(Paragraph(l, base), leftIndent=12) for l in lines]
        return ListFlowable(items, bulletType="bullet", start="•", leftIndent=14)

    def money_table(rows: list[list[str]], col_widths=None):
        if not rows:
            return Paragraph("", base)
        tbl = Table(rows, colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9.5),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))
        return tbl

    # Détail revenus encodés (ligne par ligne)
    def render_revenus_block(title: str, revenus_list: list, cfg_soc: dict):
        story.append(Paragraph(title, h3))
        if not revenus_list:
            story.append(Paragraph("Aucun revenu encodé.", base))
            return

        rows = [["Type/label", "Règle", "Montant (annuel)"]]
        for r in revenus_list:
            label = _safe(r.get("label", ""))
            typ = _safe(r.get("type", "standard"))
            a = float(r.get("montant_annuel", 0.0))
            rows.append([label, typ, f"{euro(a)} €"])
        story.append(money_table(rows, col_widths=[8.2*cm, 4.0*cm, 4.0*cm]))
        story.append(Spacer(1, 4))

        # rappel exo (info)
        story.append(Paragraph(
            f"<font size=9 color='grey'>Exo socio-pro max : {euro(cfg_soc.get('max_mensuel',0))} €/mois — "
            f"Exo artistique irrégulier : {euro(cfg_soc.get('artistique_annuel',0))} €/an</font>",
            small
        ))

    def render_cohabitants_block(cohabitants: list, res_seg: dict):
        story.append(Paragraph("Ressources des cohabitants :", h3))

        active_info = []
        for c in cohabitants or []:
            typ = normalize_art34_type(c.get("type", "autre"))
            rev_ann = float(c.get("revenu_net_annuel", 0.0))
            dq = c.get("date_quitte_menage")
            excl = bool(c.get("exclure", False))
            dq_txt = f" (départ: {date_fr(dq)})" if dq else ""
            if excl:
                active_info.append(f"{typ} — {euro(rev_ann)} €/an — EXCLU{dq_txt}")
            else:
                active_info.append(f"{typ} — {euro(rev_ann)} €/an{dq_txt}")

        if active_info:
            story.append(bullets(active_info))
        else:
            story.append(Paragraph("Aucun cohabitant encodé.", base))

        # Détail art.34 calcul
        taux = float(res_seg.get("taux_a_laisser_mensuel", 0.0))
        part_m = float(res_seg.get("cohabitants_part_a_compter_mensuel", 0.0))
        part_ann = float(res_seg.get("cohabitants_part_a_compter_annuel", 0.0))

        partn_m = float(res_seg.get("revenus_partenaire_mensuels_total", 0.0))
        debt_m_tot = float(res_seg.get("revenus_debiteurs_mensuels_total", 0.0))
        n_debt = int(res_seg.get("cohabitants_n_debiteurs_pris_en_compte", 0))
        debt_av = float(res_seg.get("cohabitants_part_debiteurs_avant_partage_mensuel", 0.0))
        debt_ap = float(res_seg.get("cohabitants_part_debiteurs_apres_partage_mensuel", 0.0))

        lines = []
        if part_m <= 0:
            lines.append("Pas de ressource cohabitant prise en compte pour la période.")
        else:
            if partn_m > 0:
                lines.append(f"Partenaire(s) : {euro(partn_m)} € (mensuel total compté)")
            if n_debt > 0:
                lines.append(f"Débiteurs : {euro(debt_m_tot)} € − {n_debt} × {euro(taux)} € = {euro(debt_av)} € (mensuel)")
                if bool(res_seg.get("partage_active", False)):
                    nshare = int(res_seg.get("nb_demandeurs_partage", 1))
                    lines.append(f"Partage : {euro(debt_av)} € / {nshare} = {euro(debt_ap)} € (mensuel)")
            lines.append(f"Total cohabitants compté : {euro(part_m)} € × 12 = {euro(part_ann)} €")
        story.append(bullets(lines))

    def render_totaux_block(res_seg: dict):
        total_av = float(res_seg.get("total_ressources_avant_immunisation_simple_annuel", 0.0))
        immu = float(res_seg.get("immunisation_simple_annuelle", 0.0))
        total_ap = float(res_seg.get("total_ressources_apres_immunisation_simple_annuel", 0.0))

        rows = [
            ["Synthèse ressources", ""],
            ["Total ressources avant immunisation", f"{euro(total_av)} €"],
            ["Immunisation simple", f"{euro(immu)} €"],
            ["Total ressources après immunisation", f"{euro(total_ap)} €"],
        ]
        tbl = Table(rows, colWidths=[10.0*cm, 6.2*cm])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("LINEBELOW", (0,0), (-1,0), 0.6, colors.black),
            ("BOX", (0,0), (-1,-1), 0.6, colors.black),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("ALIGN", (1,1), (1,-1), "RIGHT"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("TOPPADDING", (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(tbl)

    def render_ri_block(res_seg: dict, seg_info: dict | None, seg_all: dict | None):
        story.append(Paragraph("Revenu d’intégration :", h2))
        taux_ann = float(res_seg.get("taux_ris_annuel", 0.0))
        total_ap = float(res_seg.get("total_ressources_apres_immunisation_simple_annuel", 0.0))
        ri_ann = float(res_seg.get("ris_theorique_annuel", 0.0))
        ri_m = float(res_seg.get("ris_theorique_mensuel", 0.0))

        lines = [f"{euro(taux_ann)} € − {euro(total_ap)} € = {euro(ri_ann)} € par an soit {euro(ri_m)} € par mois"]
        if seg_info and seg_all:
            lines.append(f"{euro(ri_m)} € × {seg_info['jours']}/{seg_all['jours_dans_mois']} = <b>{euro(seg_info['montant_segment'])} €</b>")
        story.append(bullets(lines))

    def render_one_period(title_period: str, res_seg: dict, seg_info: dict | None, seg_all: dict | None):
        story.append(Paragraph("Calcul :", h2))
        story.append(Paragraph(title_period, h3))

        story.append(Paragraph("Ressources à considérer <font size=9>(ne pas oublier de déduire l’immunisation forfaitaire)</font> :", h2))

        # Ressources propres : détails demandeur / conjoint
        cfg_soc = engine["config"]["socio_prof"]
        story.append(Paragraph("Ressources propres :", h3))
        render_revenus_block("Revenus demandeur", answers_snapshot.get("revenus_demandeur_annuels", []), cfg_soc)
        if bool(answers_snapshot.get("couple_demandeur", False)):
            render_revenus_block("Revenus conjoint (si demande couple)", answers_snapshot.get("revenus_conjoint_annuels", []), cfg_soc)

        # Autres postes propres (capitaux / immo / cession / PF / avantage)
        cap_total = float(res_seg.get("capitaux_mobiliers_annuels", 0.0))
        immo_total = float(res_seg.get("immo_annuels", 0.0))
        ces_total = float(res_seg.get("cession_biens_annuelle", 0.0))
        pf_ann = float(res_seg.get("prestations_familiales_a_compter_annuel", 0.0))
        avn_ann = float(res_seg.get("avantage_nature_logement_annuel", 0.0))

        story.append(bullets([
            f"Capitaux mobiliers : {euro(cap_total)} € (annuel)",
            f"Immobilier : {euro(immo_total)} € (annuel)",
            f"Cession : {euro(ces_total)} € (annuel)",
            f"Prestations familiales : {euro(pf_ann)} € (annuel) [= {euro(float(res_seg.get('prestations_familiales_a_compter_mensuel',0)))} €/mois × 12]",
            f"Avantage en nature logement : {euro(avn_ann)} € (annuel) [= {euro(float(res_seg.get('avantage_nature_logement_mensuel',0)))} €/mois × 12]",
        ]))
        story.append(Spacer(1, 6))

        # Cohabitants : lignes + art.34
        render_cohabitants_block(answers_snapshot.get("cohabitants_art34", []), res_seg)
        story.append(Spacer(1, 8))

        # Totaux / immunisation
        render_totaux_block(res_seg)
        story.append(Spacer(1, 10))

        # RI
        render_ri_block(res_seg, seg_info, seg_all)
        story.append(Spacer(1, 8))

    # =========================================================
    # Corps : segments (1er mois) + mois suivants, ou mois complet
    # =========================================================
    if seg_first_month and seg_first_month.get("segments"):
        # Segments 1er mois
        for idx, s in enumerate(seg_first_month["segments"]):
            res_seg = s.get("_detail_res") if isinstance(s.get("_detail_res"), dict) else res_mois_suivants
            title_period = f"Du {date_fr(s['du'])} au {date_fr(s['au'])} :"
            render_one_period(title_period, res_seg, s, seg_first_month)

            # PageBreak après 1 segment si ça déborde (simple règle : 1 segment / page)
            # (ça fait très “décision CPAS lisible”, au lieu d’un pavé)
            if idx == 0 and len(seg_first_month["segments"]) > 1:
                story.append(PageBreak())

        # Total 1er mois
        story.append(Paragraph(f"--&gt; Soit un montant total de <b>{euro(seg_first_month.get('ris_1er_mois_total',0))} €</b> pour le mois concerné", base))
        story.append(Spacer(1, 8))

        # Mois suivants sur nouvelle page
        story.append(PageBreak())
        render_one_period("Mois suivants (situation après dernier changement dans le mois) :", res_mois_suivants, None, None)

    else:
        render_one_period("Mois complet :", res_mois_suivants, None, None)

    # Footer discret
    story.append(Spacer(1, 10))
    story.append(Paragraph("Document généré automatiquement — à valider selon la décision du CPAS.", small))

    doc.build(story)
    buf.seek(0)
    return buf


# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(page_title="Calcul RIS (CPAS officiel)", layout="centered")

if os.path.exists("logo.png"):
    st.image("logo.png", use_container_width=False)

st.title("Calcul RIS — Prototype (CPAS officiel : annuel → /12 + segments)")
st.caption("Taux RIS ANNUELS (référence centime près). Revenus encodables mensuel OU annuel. Gestion départ cohabitant (segments CPAS).")

engine = load_engine()
cfg = engine["config"]

with st.sidebar:
    st.subheader("Paramètres (JSON / indexables)")

    st.write("**Taux RIS ANNUELS (référence)** ✅")
    cfg["ris_rates_annuel"]["cohab"] = st.number_input("RIS cohab (€/an)", min_value=0.0, value=float(cfg["ris_rates_annuel"].get("cohab") or 0.0), format="%.2f")
    cfg["ris_rates_annuel"]["isole"] = st.number_input("RIS isolé (€/an)", min_value=0.0, value=float(cfg["ris_rates_annuel"].get("isole") or 0.0), format="%.2f")
    cfg["ris_rates_annuel"]["fam_charge"] = st.number_input("RIS fam. charge (€/an)", min_value=0.0, value=float(cfg["ris_rates_annuel"].get("fam_charge") or 0.0), format="%.2f")

    st.caption("Info: mensuel dérivé = annuel / 12")
    st.write(f"- cohab: {r2(cfg['ris_rates_annuel']['cohab']/12.0):.2f} €/mois")
    st.write(f"- isolé: {r2(cfg['ris_rates_annuel']['isole']/12.0):.2f} €/mois")
    st.write(f"- fam_charge: {r2(cfg['ris_rates_annuel']['fam_charge']/12.0):.2f} €/mois")

    st.divider()
    st.write("**Art.34 : taux cat.1 à laisser (€/mois)**")
    cfg["art34"]["taux_a_laisser_mensuel"] = st.number_input(
        "Taux à laisser aux débiteurs admissibles",
        min_value=0.0,
        value=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        format="%.2f"
    )

    st.divider()
    st.write("**Prestations familiales (indexables)**")
    cfg["pf"]["pf_mensuel_defaut"] = st.number_input(
        "PF (€/mois) — valeur de référence",
        min_value=0.0,
        value=float(cfg["pf"].get("pf_mensuel_defaut", 0.0)),
        format="%.2f"
    )

    st.divider()
    st.write("**Immunisation simple (€/an)**")
    cfg["immunisation_simple_annuelle"]["cohab"] = st.number_input("Immu cohab (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["cohab"]), format="%.2f")
    cfg["immunisation_simple_annuelle"]["isole"] = st.number_input("Immu isolé (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["isole"]), format="%.2f")
    cfg["immunisation_simple_annuelle"]["fam_charge"] = st.number_input("Immu fam. charge (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["fam_charge"]), format="%.2f")

    st.divider()
    st.write("**Exonérations socio-pro**")
    cfg["socio_prof"]["max_mensuel"] = st.number_input("Exo socio-pro max (€/mois)", min_value=0.0, value=float(cfg["socio_prof"]["max_mensuel"]), format="%.2f")
    cfg["socio_prof"]["artistique_annuel"] = st.number_input("Exo artistique irrégulier (€/an)", min_value=0.0, value=float(cfg["socio_prof"]["artistique_annuel"]), format="%.2f")


# ---------------------------
# UI Helpers
# ---------------------------
def ui_money_period_input(label: str, key_prefix: str, default: float = 0.0, step: float = 100.0) -> tuple[float, str]:
    c1, c2 = st.columns([1.2, 1])
    period = c1.selectbox("Période", ["Annuel (€/an)", "Mensuel (€/mois)"], key=f"{key_prefix}_period")
    if period.startswith("Annuel"):
        v = c2.number_input(label, min_value=0.0, value=float(default), step=float(step), key=f"{key_prefix}_val_a")
        return float(v), "annuel"
    else:
        v = c2.number_input(label, min_value=0.0, value=float(default), step=float(step/12.0 if step else 10.0), key=f"{key_prefix}_val_m")
        return float(v) * 12.0, "mensuel"


def ui_revenus_block(prefix: str) -> list:
    lst = []
    nb = st.number_input(f"Nombre de revenus à encoder ({prefix})", min_value=0, value=1, step=1, key=f"{prefix}_nb")
    for i in range(int(nb)):
        st.markdown(f"**Revenu {i+1} ({prefix})**")
        c1, c2, c3 = st.columns([2, 1, 1])

        label = c1.text_input("Type/label", value="salaire/chômage", key=f"{prefix}_lab_{i}")
        montant_annuel, _p = ui_money_period_input("Montant net", key_prefix=f"{prefix}_money_{i}", default=0.0, step=100.0)

        typ = c3.selectbox(
            "Règle",
            ["standard", "socio_prof", "etudiant", "artistique_irregulier", "ale", "prestations_familiales"],
            key=f"{prefix}_t_{i}"
        )

        eligible = True
        ale_part_exc_m = 0.0
        if typ in ("socio_prof", "etudiant", "artistique_irregulier"):
            eligible = st.checkbox("Éligible exonération ?", value=True, key=f"{prefix}_el_{i}")
        if typ == "ale":
            ale_part_exc_m = st.number_input("Part ALE à compter (>4,10€) (€/mois)", min_value=0.0, value=0.0, step=1.0, key=f"{prefix}_ale_{i}")

        lst.append({
            "label": label,
            "montant_annuel": float(montant_annuel),
            "type": typ,
            "eligible": eligible,
            "ale_part_excedentaire_mensuel": float(ale_part_exc_m)
        })
    return lst


def ui_menage_common(prefix: str, nb_demandeurs: int, enable_pf_links: bool) -> dict:
    answers = {}

    st.divider()
    st.subheader("Ménage (commun)")

    answers["partage_enfants_jeunes_actif"] = st.checkbox(
        "Partager la part art.34 entre plusieurs ENFANTS/JEUNES demandeurs (uniquement dans ce cas)",
        value=False,
        key=f"{prefix}_partage"
    )
    answers["nb_enfants_jeunes_demandeurs"] = 1
    if answers["partage_enfants_jeunes_actif"]:
        answers["nb_enfants_jeunes_demandeurs"] = st.number_input(
            "Nombre de demandeurs à partager",
            min_value=1, value=max(2, nb_demandeurs), step=1,
            key=f"{prefix}_nb_partage"
        )

    st.markdown("### Cohabitants admissibles (art.34) — mode simple")
    st.caption("Tu peux encoder la date de départ du ménage. Après cette date, la personne ne compte plus.")
    nb_coh = st.number_input("Nombre de cohabitants à encoder", min_value=0, value=2, step=1, key=f"{prefix}_nbcoh")

    cohabitants = []
    pf_links = []

    for i in range(int(nb_coh)):
        st.markdown(f"**Cohabitant {i+1}**")
        c1, c2, c3 = st.columns([2, 1, 1])
        typ = c1.selectbox(
            "Type",
            ["partenaire", "debiteur_direct_1", "debiteur_direct_2", "autre", "debiteur direct 1", "debiteur direct 2"],
            key=f"{prefix}_coh_t_{i}"
        )

        rev_annuel, _p = ui_money_period_input("Revenus nets", key_prefix=f"{prefix}_coh_rev_{i}", default=0.0, step=100.0)
        excl = c3.checkbox("Ne pas prendre en compte (équité / décision CPAS)", value=False, key=f"{prefix}_coh_x_{i}")

        dq = st.date_input(
            "Date de départ du ménage (optionnel) — dernier jour ensemble",
            value=None,
            key=f"{prefix}_coh_dq_{i}"
        )

        if enable_pf_links:
            c4, c5, c6 = st.columns([1.2, 1, 1])
            has_pf = c4.checkbox("PF perçues ?", value=False, key=f"{prefix}_coh_pf_yes_{i}")
            if has_pf:
                pf_m = c5.number_input("PF (€/mois)", min_value=0.0, value=0.0, step=10.0, key=f"{prefix}_coh_pf_m_{i}")
                dem_idx = c6.number_input("Pour demandeur #", min_value=1, max_value=nb_demandeurs, value=1, step=1, key=f"{prefix}_coh_pf_dem_{i}")
                pf_links.append({"dem_index": int(dem_idx) - 1, "pf_mensuel": float(pf_m)})

        cohabitants.append({
            "type": typ,
            "revenu_net_annuel": float(rev_annuel),
            "exclure": bool(excl),
            "date_quitte_menage": str(dq) if isinstance(dq, date) else None
        })

    answers["cohabitants_art34"] = cohabitants
    answers["pf_links"] = pf_links

    st.divider()
    st.markdown("### Capitaux mobiliers (ménage)")
    a_cap = st.checkbox("Le ménage possède des capitaux mobiliers", value=False, key=f"{prefix}_cap_yes")
    answers["capital_mobilier_total"] = 0.0
    answers["capital_compte_commun"] = False
    answers["capital_nb_titulaires"] = 1
    answers["capital_conjoint_cotitulaire"] = False
    answers["capital_fraction"] = 1.0
    if a_cap:
        answers["capital_mobilier_total"] = st.number_input("Montant total capitaux (€)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_cap_total")
        answers["capital_compte_commun"] = st.checkbox("Compte commun ?", value=False, key=f"{prefix}_cap_cc")
        if answers["capital_compte_commun"]:
            answers["capital_nb_titulaires"] = st.number_input("Nombre de titulaires", min_value=1, value=2, step=1, key=f"{prefix}_cap_nbtit")
        else:
            answers["capital_fraction"] = st.number_input("Part du ménage demandeur (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"{prefix}_cap_frac")

    st.divider()
    st.markdown("### Biens immobiliers (ménage)")
    biens = []
    a_immo = st.checkbox("Le ménage possède des biens immobiliers", value=False, key=f"{prefix}_immo_yes")
    if a_immo:
        nb_biens = st.number_input("Nombre de biens", min_value=0, value=1, step=1, key=f"{prefix}_immo_n")
        for i in range(int(nb_biens)):
            st.markdown(f"**Bien {i+1}**")
            habitation_principale = st.checkbox("Habitation principale ?", value=False, key=f"{prefix}_im_hp_{i}")
            bati = st.checkbox("Bien bâti ?", value=True, key=f"{prefix}_im_b_{i}")
            rc = st.number_input("RC non indexé annuel", min_value=0.0, value=0.0, step=50.0, key=f"{prefix}_im_rc_{i}")
            frac = st.number_input("Fraction droits (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"{prefix}_im_f_{i}")

            hyp = False
            interets = 0.0
            viager = False
            rente = 0.0
            if not habitation_principale:
                hyp = st.checkbox("Hypothèque ?", value=False, key=f"{prefix}_im_h_{i}")
                if hyp:
                    interets = st.number_input("Intérêts hypothécaires annuels", min_value=0.0, value=0.0, step=50.0, key=f"{prefix}_im_int_{i}")
                viager = st.checkbox("Viager ?", value=False, key=f"{prefix}_im_v_{i}")
                if viager:
                    rente = st.number_input("Rente viagère annuelle", min_value=0.0, value=0.0, step=50.0, key=f"{prefix}_im_r_{i}")

            biens.append({
                "habitation_principale": habitation_principale,
                "bati": bati,
                "rc_non_indexe": float(rc),
                "fraction_droits": float(frac),
                "hypotheque": hyp,
                "interets_annuels": float(interets),
                "viager": viager,
                "rente_viagere_annuelle": float(rente)
            })
    answers["biens_immobiliers"] = biens

    st.divider()
    st.markdown("### Cession de biens (ménage)")
    cessions = []
    a_ces = st.checkbox("Le ménage a cédé des biens (10 dernières années)", value=False, key=f"{prefix}_ces_yes")
    answers["cessions"] = []
    answers["cession_cas_particulier_37200"] = False
    answers["cession_dettes_deductibles"] = 0.0
    answers["cession_abatt_cat"] = "cat1"
    answers["cession_abatt_mois"] = 0

    if a_ces:
        answers["cession_cas_particulier_37200"] = st.checkbox("Cas particulier: tranche immunisée 37.200€", value=False, key=f"{prefix}_ces_37200")
        dettes_ok = st.checkbox("Déduire des dettes personnelles ?", value=False, key=f"{prefix}_ces_det_ok")
        if dettes_ok:
            answers["cession_dettes_deductibles"] = st.number_input("Dettes déductibles (€)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_ces_det")
        answers["cession_abatt_cat"] = st.selectbox("Catégorie d’abattement", ["cat1", "cat2", "cat3"], key=f"{prefix}_ces_cat")
        answers["cession_abatt_mois"] = st.number_input("Prorata mois", min_value=0, max_value=12, value=0, step=1, key=f"{prefix}_ces_mois")
        nb_c = st.number_input("Nombre de cessions", min_value=0, value=1, step=1, key=f"{prefix}_ces_n")
        for i in range(int(nb_c)):
            st.markdown(f"**Cession {i+1}**")
            val = st.number_input("Valeur vénale (€)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_ces_v_{i}")
            usuf = st.checkbox("Usufruit ?", value=False, key=f"{prefix}_ces_u_{i}")
            cessions.append({"valeur_venale": float(val), "usufruit": bool(usuf)})
        answers["cessions"] = cessions

    st.divider()
    st.markdown("### Avantage en nature")
    answers["avantage_nature_logement_mensuel"] = st.number_input(
        "Logement payé par un tiers non cohabitant (€/mois) — montant à compter",
        min_value=0.0, value=0.0, step=10.0,
        key=f"{prefix}_avn"
    )

    return answers


# ------------------------------------------------------------
# MODE 1 DEMANDEUR
# ------------------------------------------------------------
answers = {}

st.subheader("Profil")
answers["categorie"] = st.selectbox("Catégorie RIS", ["cohab", "isole", "fam_charge"])
answers["enfants_a_charge"] = st.number_input("Enfants à charge", min_value=0, value=0, step=1)

st.divider()
st.subheader("Date de la demande (segments CPAS)")
answers["date_demande"] = st.date_input("Date de la demande", value=date.today())

st.divider()
st.subheader("1) Revenus du demandeur — encodage mensuel OU annuel")
answers["couple_demandeur"] = st.checkbox("Demande introduite par un COUPLE (2 demandeurs ensemble)", value=False)

st.markdown("**Demandeur 1**")
answers["revenus_demandeur_annuels"] = ui_revenus_block("dem")

answers["revenus_conjoint_annuels"] = []
if answers["couple_demandeur"]:
    st.divider()
    st.markdown("**Demandeur 2 (conjoint/partenaire) — revenus à additionner**")
    answers["revenus_conjoint_annuels"] = ui_revenus_block("conj")

st.divider()
st.subheader("PF à compter (spécifiques au demandeur)")
st.caption("Astuce : si tu encodes les PF comme revenu (type 'prestations_familiales'), laisse ce champ à 0 pour éviter un double comptage.")
answers["prestations_familiales_a_compter_mensuel"] = st.number_input(
    "Prestations familiales à compter (€/mois)",
    min_value=0.0,
    value=float(cfg["pf"].get("pf_mensuel_defaut", 0.0)),
    step=10.0
)

menage = ui_menage_common("single_menage", nb_demandeurs=1, enable_pf_links=False)
answers.update(menage)

st.divider()
if st.button("Calculer le RIS"):
    seg = compute_first_month_segments(answers, engine)
    res_suiv = seg["detail_mois_suivants"]

    st.success("Calcul terminé ✅")

    st.metric("RIS mois suivants (€/mois)", f"{seg['ris_mois_suivants']:.2f}")
    st.metric("RIS du 1er mois (segments CPAS)", f"{seg['ris_1er_mois_total']:.2f}")

    st.markdown("### Détail segments du 1er mois")
    for s in seg["segments"]:
        st.write(f"- Du {s['du']} au {s['au']} : {s['ris_mensuel']:.2f} €/mois × {s['jours']}/{seg['jours_dans_mois']} = {s['montant_segment']:.2f} €")

    st.caption(f"Date demande: {seg['date_demande']} | Référence mois suivants: {seg['reference_mois_suivants']}")

    st.divider()
    st.write("### Détail (mois suivants — CPAS officiel annuel puis mensuel)")
    st.json(res_suiv)

    # PDF CPAS amélioré
    pdf_buf = make_decision_pdf_cpas(
        dossier_label="Demandeur",
        answers_snapshot=answers,      # 🔥 indispensable pour lignes par lignes
        res_mois_suivants=res_suiv,
        seg_first_month=seg,
        logo_path="logo.png"
    )
    if pdf_buf is not None:
        st.download_button(
            "⬇️ Export PDF décision (mise en page CPAS)",
            data=pdf_buf,
            file_name="decision_RI_CPAS.pdf",
            mime="application/pdf"
        )
    else:
        st.info("PDF indisponible ici (reportlab non installé). Ajoute `reportlab` dans requirements.txt.")
