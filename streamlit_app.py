 import json
import os
import calendar
from datetime import date
from io import BytesIO

import streamlit as st

# ============================================================
# CONFIG PAR DÉFAUT (fusion avec ris_rules.json si présent)
# ============================================================
DEFAULT_ENGINE = {
    "version": "1.3",
    "config": {
        # Taux RIS (mensuel)
        "ris_rates": {"cohab": 876.13, "isole": 1314.20, "fam_charge": 1776.07},

        # Option: forcer un taux annuel exact (si tes feuilles utilisent un annuel exact différent)
        "ris_rates_annuel": {"cohab": None, "isole": None, "fam_charge": None},

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

    for k in ("cohab", "isole", "fam_charge"):
        cfg["ris_rates"][k] = float(cfg["ris_rates"].get(k, 0.0))
        cfg["immunisation_simple_annuelle"][k] = float(cfg["immunisation_simple_annuelle"].get(k, 0.0))
        if "ris_rates_annuel" not in cfg:
            cfg["ris_rates_annuel"] = {"cohab": None, "isole": None, "fam_charge": None}
        if cfg["ris_rates_annuel"].get(k) is not None:
            cfg["ris_rates_annuel"][k] = float(cfg["ris_rates_annuel"][k])

    if "art34" not in cfg:
        cfg["art34"] = {}
    cfg["art34"]["taux_a_laisser_mensuel"] = float(
        cfg["art34"].get("taux_a_laisser_mensuel", cfg["ris_rates"]["cohab"])
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


def month_prorata_from_request_date(d: date) -> dict:
    days_in_month = calendar.monthrange(d.year, d.month)[1]
    days_remaining_inclusive = days_in_month - d.day + 1
    prorata = max(0.0, min(1.0, days_remaining_inclusive / days_in_month))
    return {
        "jours_dans_mois": int(days_in_month),
        "jours_restants_inclus": int(days_remaining_inclusive),
        "prorata": float(prorata),
    }


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
# REVENUS (exo socio-pro) -> encode ANNUEL, applique mensuel, remonte ANNUEL
# + ajout type "prestations_familiales"
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
            # PF encodées comme revenu annuel (comptées "standard")
            total_m += m
        else:
            total_m += m
    return float(max(0.0, total_m * 12.0))


# ============================================================
# ART.34 — MODE SIMPLE (comme ton code d'origine amélioré)
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


ADMISSIBLES_ART34 = {"partenaire", "debiteur_direct_1", "debiteur_direct_2"}


def cohabitants_art34_part_mensuelle_cpas(cohabitants: list,
                                         taux_a_laisser_mensuel: float,
                                         partage_active: bool,
                                         nb_demandeurs_a_partager: int) -> dict:
    taux = max(0.0, float(taux_a_laisser_mensuel))

    revenus_debiteurs_m = 0.0
    nb_debiteurs = 0

    for c in cohabitants:
        typ = normalize_art34_type(c.get("type", "autre"))
        if typ not in ADMISSIBLES_ART34:
            continue
        if bool(c.get("exclure", False)):
            continue

        revenu_ann = max(0.0, float(c.get("revenu_net_annuel", 0.0)))
        revenu_m = revenu_ann / 12.0

        revenus_debiteurs_m += revenu_m
        nb_debiteurs += 1  # IMPORTANT: même si revenu = 0

    part_m = max(0.0, revenus_debiteurs_m - (nb_debiteurs * taux))

    if partage_active:
        n = max(1, int(nb_demandeurs_a_partager))
        part_m_par_dem = part_m / n
    else:
        part_m_par_dem = part_m

    part_m = r2(part_m)
    part_m_par_dem = r2(part_m_par_dem)

    return {
        "cohabitants_n_pris_en_compte": int(nb_debiteurs),
        "revenus_debiteurs_mensuels_total": r2(revenus_debiteurs_m),
        "cohabitants_part_totale_avant_partage_mensuel": part_m,
        "cohabitants_part_a_compter_mensuel": part_m_par_dem,
        "cohabitants_part_a_compter_annuel": r2(part_m_par_dem * 12.0),
    }


# ============================================================
# ART.34 — MENAGE AVANCE (pool + priorité 1er/2e degré + partage)
# ============================================================
def make_pool_key(ids: list) -> str:
    a = ",".join(sorted([str(x) for x in (ids or []) if str(x).strip()]))
    return f"ids[{a}]"


def art34_group_excess_m(debtors: list, taux: float, extra_income_m: float = 0.0) -> float:
    n = len(debtors)
    s = sum(max(0.0, float(d.get("revenu_net_annuel", 0.0))) / 12.0 for d in debtors) + max(0.0, float(extra_income_m))
    return r2(max(0.0, s - (n * float(taux))))


def art34_draw_from_pool(degree: int,
                         debtor_ids: list,
                         household: dict,
                         taux: float,
                         pools: dict,
                         share_plan: dict,
                         include_ris_m: float,
                         include_ris_from: list) -> dict:
    ids = list(debtor_ids or [])
    debtors = [household["members_by_id"][i] for i in ids if i in household["members_by_id"]]

    key = make_pool_key(ids)
    base = art34_group_excess_m(debtors, taux, extra_income_m=include_ris_m)

    if key not in pools:
        pools[key] = float(base)

    if key in share_plan and share_plan[key]["count"] > 1:
        per = float(share_plan[key]["per"])
        take = min(pools[key], per)
    else:
        take = pools[key]

    take = r2(max(0.0, take))
    pools[key] = r2(max(0.0, pools[key] - take))

    return {
        "key": key,
        "degree": degree,
        "nb_debiteurs": len(debtors),
        "revenus_m_total_avec_injections": r2(
            sum(max(0.0, float(d.get("revenu_net_annuel", 0.0))) / 12.0 for d in debtors) + include_ris_m
        ),
        "base_exces_m": float(base),
        "pris_en_compte_m": float(take),
        "reste_pool_m": float(pools[key]),
    }


def compute_art34_menage_avance(dossier: dict,
                               household: dict,
                               taux: float,
                               pools: dict,
                               share_plan: dict,
                               prior_results: list) -> dict:
    include_from = dossier.get("include_ris_from_dossiers", []) or []
    include_ris_m = 0.0
    for idx in include_from:
        if 0 <= idx < len(prior_results) and prior_results[idx] is not None:
            include_ris_m += float(prior_results[idx].get("ris_theorique_mensuel", 0.0))
    include_ris_m = r2(include_ris_m)

    deg1_ids = dossier.get("art34_deg1_ids", []) or []
    deg2_ids = dossier.get("art34_deg2_ids", []) or []

    dbg1 = art34_draw_from_pool(
        degree=1, debtor_ids=deg1_ids, household=household, taux=taux,
        pools=pools, share_plan=share_plan, include_ris_m=include_ris_m,
        include_ris_from=include_from
    ) if len(deg1_ids) > 0 else None

    part_m = float(dbg1["pris_en_compte_m"]) if dbg1 else 0.0
    used_degree = 1 if part_m > 0 else 0

    dbg2 = None
    if part_m <= 0.0 and len(deg2_ids) > 0:
        dbg2 = art34_draw_from_pool(
            degree=2, debtor_ids=deg2_ids, household=household, taux=taux,
            pools=pools, share_plan=share_plan, include_ris_m=0.0,
            include_ris_from=[]
        )
        part_m = float(dbg2["pris_en_compte_m"])
        used_degree = 2 if part_m > 0 else 0

    part_m = r2(part_m)
    return {
        "art34_mode": "MENAGE_AVANCE",
        "art34_degree_utilise": used_degree,
        "cohabitants_part_a_compter_mensuel": float(part_m),
        "cohabitants_part_a_compter_annuel": float(r2(part_m * 12.0)),
        "debug_deg1": dbg1,
        "debug_deg2": dbg2,
        "ris_injecte_mensuel": float(include_ris_m),
    }


# ============================================================
# CALCUL GLOBAL — OFFICIEL CPAS (ANNUEL puis /12)
# ============================================================
def compute_officiel_cpas_annuel(answers: dict, engine: dict) -> dict:
    cfg = engine["config"]
    cat = answers.get("categorie", "isole")

    taux_ris_m = float(cfg["ris_rates"].get(cat, 0.0))
    if cfg.get("ris_rates_annuel", {}).get(cat) is not None:
        taux_ris_annuel = float(cfg["ris_rates_annuel"][cat])
    else:
        taux_ris_annuel = r2(taux_ris_m) * 12.0
        taux_ris_annuel = r2(taux_ris_annuel)

    # 1) Revenus demandeur (+ conjoint si couple-demandeur)
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

    # 2) Capitaux
    cap_ann = capital_mobilier_annuel(
        total_capital=answers.get("capital_mobilier_total", 0.0),
        compte_commun=answers.get("capital_compte_commun", False),
        nb_titulaires=answers.get("capital_nb_titulaires", 1),
        categorie=cat,
        conjoint_compte_commun=answers.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=answers.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    )
    cap_ann = r2(cap_ann)

    # 3) Immo
    immo_ann = immo_annuel_total(
        biens=answers.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )
    immo_ann = r2(immo_ann)

    # 4) Cession
    ces_ann = cession_biens_annuelle(
        cessions=answers.get("cessions", []),
        cas_particulier_tranche_37200=answers.get("cession_cas_particulier_37200", False),
        dettes_deductibles=answers.get("cession_dettes_deductibles", 0.0),
        abatt_cat=answers.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=answers.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    )
    ces_ann = r2(ces_ann)

    # 5) Art.34 (simple)
    art34 = cohabitants_art34_part_mensuelle_cpas(
        cohabitants=answers.get("cohabitants_art34", []),
        taux_a_laisser_mensuel=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        partage_active=bool(answers.get("partage_enfants_jeunes_actif", False)),
        nb_demandeurs_a_partager=int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
    )

    # 5bis) PF (mensuel -> annuel)
    pf_m = max(0.0, float(answers.get("prestations_familiales_a_compter_mensuel", 0.0)))
    pf_m = r2(pf_m)
    pf_ann = r2(pf_m * 12.0)

    # 6) Avantage en nature logement (mensuel -> annuel)
    avantage_nature_m = max(0.0, float(answers.get("avantage_nature_logement_mensuel", 0.0)))
    avantage_nature_m = r2(avantage_nature_m)
    avantage_nature_ann = r2(avantage_nature_m * 12.0)

    total_avant_annuel = (
        revenus_demandeur_annuels
        + cap_ann
        + immo_ann
        + ces_ann
        + art34["cohabitants_part_a_compter_annuel"]
        + pf_ann
        + avantage_nature_ann
    )
    total_avant_annuel = r2(total_avant_annuel)

    # Immunisation simple (annuelle) si ressources < taux
    immu_ann = 0.0
    if taux_ris_annuel > 0 and total_avant_annuel < taux_ris_annuel:
        immu_ann = float(cfg["immunisation_simple_annuelle"].get(cat, 0.0))
    immu_ann = r2(immu_ann)

    total_apres_annuel = max(0.0, total_avant_annuel - immu_ann)
    total_apres_annuel = r2(total_apres_annuel)

    ris_annuel = max(0.0, taux_ris_annuel - total_apres_annuel) if taux_ris_annuel > 0 else 0.0
    ris_annuel = r2(ris_annuel)
    ris_mensuel = r2(ris_annuel / 12.0)

    # Prorata 1er mois
    d_dem = answers.get("date_demande", date.today())
    pr = month_prorata_from_request_date(d_dem)
    ris_premier_mois = r2(ris_mensuel * pr["prorata"])

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

        "taux_ris_mensuel": float(r2(taux_ris_m)),
        "ris_theorique_mensuel": float(ris_mensuel),

        "date_demande": str(d_dem),
        "jours_dans_mois": pr["jours_dans_mois"],
        "jours_restants_inclus": pr["jours_restants_inclus"],
        "prorata_premier_mois": pr["prorata"],
        "ris_premier_mois_prorata": float(ris_premier_mois),
        "ris_mois_suivants": float(ris_mensuel),
    }


# ============================================================
# EXPORT PDF
# ============================================================
def build_decision_text(dossier_label: str, res: dict) -> str:
    lines = []
    lines.append(f"Décision RIS — {dossier_label}")
    lines.append("")
    lines.append(f"Catégorie: {res['categorie']}")
    lines.append(f"Taux RIS mensuel: {res['taux_ris_mensuel']:.2f} €")
    lines.append(f"Taux RIS annuel: {res['taux_ris_annuel']:.2f} €")
    lines.append("")
    lines.append("Ressources (annuel):")
    lines.append(f"- Revenus demandeur: {res['revenus_demandeur_annuels']:.2f} €")
    lines.append(f"- Capitaux mobiliers: {res['capitaux_mobiliers_annuels']:.2f} €")
    lines.append(f"- Immobilier: {res['immo_annuels']:.2f} €")
    lines.append(f"- Cession: {res['cession_biens_annuelle']:.2f} €")
    lines.append(f"- Art.34 (annuel): {res['cohabitants_part_a_compter_annuel']:.2f} €")
    lines.append(f"- PF à compter (annuel): {res['prestations_familiales_a_compter_annuel']:.2f} €")
    lines.append(f"- Avantage nature logement (annuel): {res['avantage_nature_logement_annuel']:.2f} €")
    lines.append("")
    lines.append(f"Total ressources avant immunisation: {res['total_ressources_avant_immunisation_simple_annuel']:.2f} €")
    lines.append(f"Immunisation simple: {res['immunisation_simple_annuelle']:.2f} €")
    lines.append(f"Total ressources après immunisation: {res['total_ressources_apres_immunisation_simple_annuel']:.2f} €")
    lines.append("")
    lines.append(f"RIS annuel: {res['ris_theorique_annuel']:.2f} €")
    lines.append(f"RIS mensuel: {res['ris_theorique_mensuel']:.2f} €")
    lines.append("")
    lines.append(f"Date demande: {res['date_demande']}")
    lines.append(f"Prorata 1er mois: {res['jours_restants_inclus']}/{res['jours_dans_mois']} = {res['prorata_premier_mois']:.6f}")
    lines.append(f"RIS 1er mois prorata: {res['ris_premier_mois_prorata']:.2f} €")
    return "\n".join(lines)


def try_make_pdf_from_text(text: str) -> BytesIO | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception:
        return None

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    x = 40
    y = height - 50
    line_h = 14

    for line in text.splitlines():
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(x, y, line[:1200])
        y -= line_h

    c.save()
    buf.seek(0)
    return buf


# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(page_title="Calcul RIS (CPAS officiel)", layout="centered")

if os.path.exists("logo.png"):
    st.image("logo.png", use_container_width=False)

st.title("Calcul RIS — Prototype (CPAS officiel : annuel puis /12)")
st.caption("Mode 'feuilles CPAS' : annuel → immunisation simple → RIS annuel → /12 + prorata 1er mois.")

engine = load_engine()
cfg = engine["config"]

with st.sidebar:
    st.subheader("Paramètres (JSON / indexables)")

    st.write("**Taux RIS (mensuels)**")
    cfg["ris_rates"]["cohab"] = st.number_input("RIS cohab (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["cohab"]), format="%.2f")
    cfg["ris_rates"]["isole"] = st.number_input("RIS isolé (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["isole"]), format="%.2f")
    cfg["ris_rates"]["fam_charge"] = st.number_input("RIS fam. charge (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["fam_charge"]), format="%.2f")

    st.divider()
    force_ann = st.checkbox("Forcer les taux RIS annuels (au lieu de mensuel×12)", value=False)
    if force_ann:
        cfg["ris_rates_annuel"]["cohab"] = st.number_input("RIS cohab (€/an)", min_value=0.0, value=float(cfg["ris_rates_annuel"].get("cohab") or 0.0), format="%.2f")
        cfg["ris_rates_annuel"]["isole"] = st.number_input("RIS isolé (€/an)", min_value=0.0, value=float(cfg["ris_rates_annuel"].get("isole") or 0.0), format="%.2f")
        cfg["ris_rates_annuel"]["fam_charge"] = st.number_input("RIS fam. charge (€/an)", min_value=0.0, value=float(cfg["ris_rates_annuel"].get("fam_charge") or 0.0), format="%.2f")
    else:
        cfg["ris_rates_annuel"] = {"cohab": None, "isole": None, "fam_charge": None}

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
# Blocs UI
# ---------------------------
def ui_revenus_annuels_block(prefix: str) -> list:
    lst = []
    nb = st.number_input(f"Nombre de revenus à encoder ({prefix})", min_value=0, value=1, step=1, key=f"{prefix}_nb")
    for i in range(int(nb)):
        st.markdown(f"**Revenu {i+1} ({prefix})**")
        c1, c2, c3 = st.columns([2, 1, 1])
        label = c1.text_input("Type/label", value="salaire/chômage", key=f"{prefix}_lab_{i}")
        montant_a = c2.number_input("Montant net ANNUEL (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_a_{i}")
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
            "montant_annuel": float(montant_a),
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
    st.caption("Débiteurs admissibles comptés même à 0€ (feuilles CPAS).")
    nb_coh = st.number_input("Nombre de cohabitants à encoder", min_value=0, value=2, step=1, key=f"{prefix}_nbcoh")

    cohabitants = []
    pf_links = []

    for i in range(int(nb_coh)):
        st.markdown(f"**Cohabitant {i+1}**")
        c1, c2 = st.columns([2, 1])
        typ = c1.selectbox(
            "Type",
            ["partenaire", "debiteur_direct_1", "debiteur_direct_2", "autre", "debiteur direct 1", "debiteur direct 2"],
            key=f"{prefix}_coh_t_{i}"
        )
        rev = c2.number_input("Revenus nets annuels (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_coh_r_{i}")
        excl = st.checkbox("Ne pas prendre en compte (équité / décision CPAS)", value=False, key=f"{prefix}_coh_x_{i}")

        if enable_pf_links:
            c3, c4, c5 = st.columns([1.2, 1, 1])
            has_pf = c3.checkbox("PF perçues ?", value=False, key=f"{prefix}_coh_pf_yes_{i}")
            if has_pf:
                pf_m = c4.number_input("PF (€/mois)", min_value=0.0, value=0.0, step=10.0, key=f"{prefix}_coh_pf_m_{i}")
                dem_idx = c5.number_input("Pour demandeur #", min_value=1, max_value=nb_demandeurs, value=1, step=1, key=f"{prefix}_coh_pf_dem_{i}")
                pf_links.append({"dem_index": int(dem_idx) - 1, "pf_mensuel": float(pf_m)})

        cohabitants.append({"type": typ, "revenu_net_annuel": float(rev), "exclure": bool(excl)})

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
# MODE DOSSIER
# ------------------------------------------------------------
st.subheader("Mode dossier")
multi_mode = st.checkbox("Plusieurs demandes RIS — comparer / calculer un ménage", value=False)

# ------------------------------------------------------------
# MODE MULTI + MENAGE AVANCE
# ------------------------------------------------------------
if multi_mode:
    st.subheader("Choix du mode multi")
    advanced_household = st.checkbox(
        "Ménage avancé (feuilles CPAS) : parents/couple + enfants + autres demandeurs + priorité + pools",
        value=True
    )

    nb_dem = st.number_input("Nombre de dossiers/demandes à calculer", min_value=2, max_value=4, value=3, step=1)

    # -----------------------
    # A) Encodage des dossiers
    # -----------------------
    st.subheader("A) Dossiers / demandes")
    dossiers = []
    for i in range(int(nb_dem)):
        st.markdown(f"### Dossier {i+1}")
        label = st.text_input("Nom/Label", value=f"Dossier {i+1}", key=f"hd_lab_{i}")
        cat = st.selectbox("Catégorie RIS", ["cohab", "isole", "fam_charge"], key=f"hd_cat_{i}")
        enfants = st.number_input("Enfants à charge", min_value=0, value=0, step=1, key=f"hd_enf_{i}")
        d_dem = st.date_input("Date de demande", value=date.today(), key=f"hd_date_{i}")

        is_couple = st.checkbox("Dossier COUPLE (2 demandeurs ensemble)", value=False, key=f"hd_couple_{i}")

        st.markdown("**Revenus nets ANNUELS (demandeur 1)**")
        rev1 = ui_revenus_annuels_block(f"hd_rev1_{i}")

        rev2 = []
        if is_couple:
            st.markdown("**Revenus nets ANNUELS (demandeur 2 / conjoint)**")
            rev2 = ui_revenus_annuels_block(f"hd_rev2_{i}")

        st.markdown("**PF à compter (spécifiques à CE dossier)**")
        st.caption("Astuce : si tu encodes les PF comme revenu annuel (type 'prestations_familiales'), laisse ce champ à 0 pour éviter un double comptage.")
        pf_m = st.number_input(
            "PF à compter (€/mois)",
            min_value=0.0,
            value=float(cfg["pf"].get("pf_mensuel_defaut", 0.0)),
            step=10.0,
            key=f"hd_pf_{i}"
        )

        share_art34 = st.checkbox(
            "Enfant/Jeune demandeur : partager la part art.34 (si même groupe débiteurs) avec les autres dossiers marqués",
            value=False,
            key=f"hd_share_{i}"
        )

        dossiers.append({
            "idx": i,
            "label": label,
            "categorie": cat,
            "enfants_a_charge": int(enfants),
            "date_demande": d_dem,
            "couple_demandeur": bool(is_couple),
            "revenus_demandeur_annuels": rev1,
            "revenus_conjoint_annuels": rev2,
            "prestations_familiales_a_compter_mensuel": float(pf_m),
            "share_art34": bool(share_art34),
            "art34_deg1_ids": [],
            "art34_deg2_ids": [],
            "include_ris_from_dossiers": [],
        })

    # -----------------------
    # B) Ménage commun (capitaux/immo/cession/avantage + PF links)
    # -----------------------
    st.subheader("B) Ménage (commun)")
    menage_common = ui_menage_common("hd_menage", nb_demandeurs=int(nb_dem), enable_pf_links=True)

    # Inject PF-links vers le bon dossier
    for link in menage_common.get("pf_links", []):
        idx = int(link["dem_index"])
        if 0 <= idx < len(dossiers):
            dossiers[idx]["prestations_familiales_a_compter_mensuel"] += float(link["pf_mensuel"])

    # -----------------------
    # C) Ménage avancé: membres + mapping des débiteurs par degré
    # -----------------------
    household = {"members": [], "members_by_id": {}}

    if advanced_household:
        st.divider()
        st.subheader("C) Ménage avancé — Membres & débiteurs (art.34)")
        st.caption("Encode ici les débiteurs potentiels (parents, grands-parents, etc.). Puis, pour chaque dossier, sélectionne le 1er et le 2e degré.")

        nb_m = st.number_input("Nombre de membres (débit. potentiels) à encoder", min_value=0, value=3, step=1)
        members = []
        for j in range(int(nb_m)):
            st.markdown(f"**Membre {j+1}**")
            c1, c2, c3 = st.columns([2, 1, 1])
            mid = c1.text_input("ID court (ex: X, Y, E)", value=f"M{j+1}", key=f"mem_id_{j}")
            name = c1.text_input("Nom (optionnel)", value="", key=f"mem_name_{j}")
            rev_a = c2.number_input("Revenus nets annuels (€/an) à prendre en compte", min_value=0.0, value=0.0, step=100.0, key=f"mem_rev_{j}")
            excl = c3.checkbox("Exclure (équité)", value=False, key=f"mem_excl_{j}")
            m = {
                "id": str(mid).strip(),
                "name": str(name).strip(),
                "revenu_net_annuel": float(rev_a),
                "exclure": bool(excl)
            }
            if m["id"]:
                members.append(m)

        members_by_id = {m["id"]: m for m in members if not m.get("exclure", False)}
        household = {"members": members, "members_by_id": members_by_id}

        ids_available = list(members_by_id.keys())

        st.divider()
        st.subheader("D) Paramétrage art.34 par dossier (degrés + injections)")
        for d in dossiers:
            st.markdown(f"### {d['label']} — art.34")
            c1, c2 = st.columns(2)
            d["art34_deg1_ids"] = c1.multiselect(
                "Débiteurs 1er degré (prioritaires)",
                options=ids_available,
                default=[],
                key=f"d_{d['idx']}_deg1"
            )
            d["art34_deg2_ids"] = c2.multiselect(
                "Débiteurs 2e degré (si 1er degré = 0 disponible)",
                options=ids_available,
                default=[],
                key=f"d_{d['idx']}_deg2"
            )

            d["include_ris_from_dossiers"] = st.multiselect(
                "Option : ajouter le RI mensuel d’autres dossiers dans les revenus des débiteurs 1er degré (cas E11 : +RI parents)",
                options=[i for i in range(len(dossiers))],
                format_func=lambda k: f"{k+1} — {dossiers[k]['label']}",
                default=[],
                key=f"d_{d['idx']}_risinj"
            )

    st.divider()
    if st.button("Calculer (multi)"):
        taux_art34 = float(cfg["art34"]["taux_a_laisser_mensuel"])

        # plan de partage (cas E6)
        share_plan = {}

        if advanced_household:
            for d in dossiers:
                if not d.get("share_art34", False):
                    continue
                ids = list(d.get("art34_deg1_ids", []) or [])
                if not ids:
                    continue
                key = make_pool_key(ids)
                if key not in share_plan:
                    share_plan[key] = {"count": 0, "per": 0.0}
                share_plan[key]["count"] += 1

            for key, v in list(share_plan.items()):
                try:
                    ids_str = key.replace("ids[", "").replace("]", "")
                    ids = [x for x in ids_str.split(",") if x]
                except Exception:
                    ids = []
                debtors = [household["members_by_id"][i] for i in ids if i in household["members_by_id"]]
                base = art34_group_excess_m(debtors, taux_art34, extra_income_m=0.0)
                if v["count"] > 0:
                    v["per"] = r2(float(base) / float(v["count"]))

        prior_results = [None] * len(dossiers)

        for _iter in range(4):
            pools = {}
            results_tmp = [None] * len(dossiers)

            for d in dossiers:
                answers = dict(menage_common)
                answers.update({
                    "categorie": d["categorie"],
                    "enfants_a_charge": d["enfants_a_charge"],
                    "date_demande": d["date_demande"],
                    "couple_demandeur": d["couple_demandeur"],
                    "revenus_demandeur_annuels": d["revenus_demandeur_annuels"],
                    "revenus_conjoint_annuels": d["revenus_conjoint_annuels"],
                    "prestations_familiales_a_compter_mensuel": d["prestations_familiales_a_compter_mensuel"],
                })

                if advanced_household:
                    art34_adv = compute_art34_menage_avance(
                        dossier=d,
                        household=household,
                        taux=taux_art34,
                        pools=pools,
                        share_plan=share_plan,
                        prior_results=prior_results
                    )

                    res = compute_officiel_cpas_annuel(answers, engine)

                    res["art34_mode"] = art34_adv["art34_mode"]
                    res["art34_degree_utilise"] = art34_adv["art34_degree_utilise"]
                    res["ris_injecte_mensuel"] = art34_adv["ris_injecte_mensuel"]
                    res["debug_art34_deg1"] = art34_adv["debug_deg1"]
                    res["debug_art34_deg2"] = art34_adv["debug_deg2"]

                    res["cohabitants_part_a_compter_mensuel"] = art34_adv["cohabitants_part_a_compter_mensuel"]
                    res["cohabitants_part_a_compter_annuel"] = art34_adv["cohabitants_part_a_compter_annuel"]

                    total_avant = (
                        float(res["revenus_demandeur_annuels"])
                        + float(res["capitaux_mobiliers_annuels"])
                        + float(res["immo_annuels"])
                        + float(res["cession_biens_annuelle"])
                        + float(res["cohabitants_part_a_compter_annuel"])
                        + float(res["prestations_familiales_a_compter_annuel"])
                        + float(res["avantage_nature_logement_annuel"])
                    )
                    total_avant = r2(total_avant)

                    taux_ris_annuel = float(res["taux_ris_annuel"])
                    immu = 0.0
                    if taux_ris_annuel > 0 and total_avant < taux_ris_annuel:
                        immu = float(cfg["immunisation_simple_annuelle"].get(res["categorie"], 0.0))
                    immu = r2(immu)

                    total_apres = r2(max(0.0, total_avant - immu))
                    ris_ann = r2(max(0.0, taux_ris_annuel - total_apres))
                    ris_m = r2(ris_ann / 12.0)

                    pr = month_prorata_from_request_date(d["date_demande"])
                    ris_1 = r2(ris_m * pr["prorata"])

                    res["total_ressources_avant_immunisation_simple_annuel"] = float(total_avant)
                    res["immunisation_simple_annuelle"] = float(immu)
                    res["total_ressources_apres_immunisation_simple_annuel"] = float(total_apres)
                    res["ris_theorique_annuel"] = float(ris_ann)
                    res["ris_theorique_mensuel"] = float(ris_m)
                    res["ris_premier_mois_prorata"] = float(ris_1)
                    res["ris_mois_suivants"] = float(ris_m)
                    res["jours_dans_mois"] = pr["jours_dans_mois"]
                    res["jours_restants_inclus"] = pr["jours_restants_inclus"]
                    res["prorata_premier_mois"] = pr["prorata"]

                else:
                    res = compute_officiel_cpas_annuel(answers, engine)

                res["_label"] = d["label"]
                res["_idx"] = d["idx"]
                results_tmp[d["idx"]] = res

            changed = False
            for i in range(len(dossiers)):
                old = prior_results[i]["ris_theorique_mensuel"] if prior_results[i] else None
                new = results_tmp[i]["ris_theorique_mensuel"] if results_tmp[i] else None
                if old is None or new is None or abs(float(old) - float(new)) > 0.005:
                    changed = True

            prior_results = results_tmp
            if not changed:
                break

        results = prior_results

        st.success("Calcul terminé ✅")

        st.markdown("## Tableau comparatif")
        table_rows = []
        for r in results:
            row = {
                "Dossier": r["_label"],
                "Catégorie": r["categorie"],
                "Couple ?": "Oui" if r.get("couple_demandeur") else "Non",
                "RIS mensuel": round(r["ris_theorique_mensuel"], 2),
                "RIS 1er mois": round(r["ris_premier_mois_prorata"], 2),
                "Art.34 mensuel compté": round(r["cohabitants_part_a_compter_mensuel"], 2),
                "PF mensuel compté": round(r["prestations_familiales_a_compter_mensuel"], 2),
                "Total ressources (annuel)": round(r["total_ressources_avant_immunisation_simple_annuel"], 2),
            }
            if advanced_household:
                row["Art.34 mode"] = r.get("art34_mode", "")
                row["Degré utilisé"] = r.get("art34_degree_utilise", 0)
                row["RI injecté (€/mois)"] = round(r.get("ris_injecte_mensuel", 0.0), 2)
            table_rows.append(row)

        st.dataframe(table_rows, use_container_width=True)

        st.divider()
        st.markdown("## Détails (par dossier)")
        for r in results:
            with st.expander(f"Détail — {r['_label']}"):
                st.metric("RIS mensuel", f"{r['ris_theorique_mensuel']:.2f} €")
                st.metric("RIS 1er mois (prorata)", f"{r['ris_premier_mois_prorata']:.2f} €")
                if advanced_household:
                    st.caption("Art.34 avancé :")
                    st.write({
                        "mode": r.get("art34_mode"),
                        "degre": r.get("art34_degree_utilise"),
                        "ri_injecte_m": r.get("ris_injecte_mensuel", 0.0),
                        "deg1": r.get("debug_art34_deg1"),
                        "deg2": r.get("debug_art34_deg2"),
                    })

                st.json(r)

                decision_txt = build_decision_text(r["_label"], r)
                st.download_button(
                    "⬇️ Export TEXTE décision",
                    data=decision_txt.encode("utf-8"),
                    file_name=f"decision_RIS_{r['_label']}.txt",
                    mime="text/plain",
                    key=f"dl_txt_{r['_label']}"
                )
                pdf_buf = try_make_pdf_from_text(decision_txt)
                if pdf_buf is not None:
                    st.download_button(
                        "⬇️ Export PDF décision",
                        data=pdf_buf,
                        file_name=f"decision_RIS_{r['_label']}.pdf",
                        mime="application/pdf",
                        key=f"dl_pdf_{r['_label']}"
                    )
                else:
                    st.info("PDF indisponible ici (reportlab non installé). Ajoute `reportlab` dans requirements.txt.")


# ------------------------------------------------------------
# MODE 1 DEMANDEUR (ton flux d’origine)
# ------------------------------------------------------------
else:
    answers = {}

    st.subheader("Profil")
    answers["categorie"] = st.selectbox("Catégorie RIS", ["cohab", "isole", "fam_charge"])
    answers["enfants_a_charge"] = st.number_input("Enfants à charge", min_value=0, value=0, step=1)

    st.divider()
    st.subheader("Date de la demande (pour calcul du 1er mois)")
    answers["date_demande"] = st.date_input("Date de la demande", value=date.today())

    st.divider()
    st.subheader("1) Revenus du demandeur — ANNUELS (nets)")
    answers["couple_demandeur"] = st.checkbox("Demande introduite par un COUPLE (2 demandeurs ensemble)", value=False)

    st.markdown("**Demandeur 1**")
    answers["revenus_demandeur_annuels"] = ui_revenus_annuels_block("dem")

    answers["revenus_conjoint_annuels"] = []
    if answers["couple_demandeur"]:
        st.divider()
        st.markdown("**Demandeur 2 (conjoint/partenaire) — revenus à additionner**")
        answers["revenus_conjoint_annuels"] = ui_revenus_annuels_block("conj")

    st.divider()
    st.subheader("PF à compter (spécifiques au demandeur)")
    st.caption("Astuce : si tu encodes les PF comme revenu annuel (type 'prestations_familiales'), laisse ce champ à 0 pour éviter un double comptage.")
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
        res = compute_officiel_cpas_annuel(answers, engine)
        st.success("Calcul terminé ✅")
        st.metric("RIS mensuel normal (€/mois)", f"{res['ris_theorique_mensuel']:.2f}")
        st.metric("RIS du 1er mois (prorata)", f"{res['ris_premier_mois_prorata']:.2f}")

        st.caption(
            f"Prorata = {res['jours_restants_inclus']}/{res['jours_dans_mois']} "
            f"= {res['prorata_premier_mois']:.6f} (jour de demande inclus)"
        )

        st.write("### Détail (CPAS officiel — annuel puis mensuel)")
        st.json(res)

        decision_txt = build_decision_text("Demandeur", res)
        st.download_button(
            "⬇️ Export TEXTE décision",
            data=decision_txt.encode("utf-8"),
            file_name="decision_RIS.txt",
            mime="text/plain"
        )
        pdf_buf = try_make_pdf_from_text(decision_txt)
        if pdf_buf is not None:
            st.download_button(
                "⬇️ Export PDF décision",
                data=pdf_buf,
                file_name="decision_RIS.pdf",
                mime="application/pdf"
            )
        else:
            st.info("PDF indisponible ici (reportlab non installé). Ajoute `reportlab` dans requirements.txt.")
