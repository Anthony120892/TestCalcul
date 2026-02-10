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
    "version": "1.0",
    "config": {
        # Taux RIS (mensuel)
        "ris_rates": {"cohab": 876.13, "isole": 1314.20, "fam_charge": 1776.07},

        # Immunisation simple (annuelle)
        "immunisation_simple_annuelle": {"cohab": 155.0, "isole": 250.0, "fam_charge": 310.0},

        # Art. 34 : taux "catégorie 1 à laisser" (mensuel) pour cohabitants admissibles
        "art34": {"taux_a_laisser_mensuel": 876.13},

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
            "max_mensuel": 274.82,
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

    if "art34" not in cfg:
        cfg["art34"] = {}
    cfg["art34"]["taux_a_laisser_mensuel"] = float(
        cfg["art34"].get("taux_a_laisser_mensuel", cfg["ris_rates"]["cohab"])
    )
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
    r1 = float(cfg_cap["t1_rate"])
    r2 = float(cfg_cap["t2_rate"])

    if adj_total <= t0_max:
        return 0.0

    annuel = 0.0
    tranche1 = max(0.0, min(adj_total, t1_max) - t1_min)
    annuel += tranche1 * r1
    tranche2 = max(0.0, adj_total - t1_max)
    annuel += tranche2 * r2
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
    r1 = float(cfg_cap["t1_rate"])
    r2 = float(cfg_cap["t2_rate"])

    if total <= t0_max:
        return 0.0

    annuel = 0.0
    tranche1 = max(0.0, min(total, t1_max) - t1_min)
    annuel += tranche1 * r1
    tranche2 = max(0.0, total - t1_max)
    annuel += tranche2 * r2
    return float(max(0.0, annuel))


# ============================================================
# REVENUS (exo socio-pro) -> encode ANNUEL, applique mensuel, remonte ANNUEL
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
        else:
            total_m += m
    return float(max(0.0, total_m * 12.0))


# ============================================================
# ART. 34 — VERSION "FEUILLES CPAS" (CORRIGÉE)
# - On somme les revenus mensuels des débiteurs admissibles
# - On retire nb_debiteurs * taux_a_laisser
# - On partage ensuite si plusieurs enfants/jeunes demandeurs
# ============================================================
ADMISSIBLES_ART34 = {"partenaire", "debiteur_direct_1", "debiteur_direct_2"}

def cohabitants_art34_part_mensuelle_cpas(cohabitants: list,
                                         taux_a_laisser_mensuel: float,
                                         partage_active: bool,
                                         nb_demandeurs_a_partager: int) -> dict:
    taux = max(0.0, float(taux_a_laisser_mensuel))

    revenus_debiteurs_m = 0.0
    nb_debiteurs = 0

    # PF "à compter" restent une autre ligne (et seront gérées par demandeur dans le mode multi)
    for c in cohabitants:
        typ = c.get("type", "autre")
        if typ not in ADMISSIBLES_ART34:
            continue
        if bool(c.get("exclure", False)):
            continue

        revenu_ann = max(0.0, float(c.get("revenu_net_annuel", 0.0)))
        revenu_m = revenu_ann / 12.0

        revenus_debiteurs_m += revenu_m
        nb_debiteurs += 1

    # Règle CPAS
    part_m = max(0.0, revenus_debiteurs_m - (nb_debiteurs * taux))

    if partage_active:
        n = max(1, int(nb_demandeurs_a_partager))
        part_m_par_dem = part_m / n
    else:
        part_m_par_dem = part_m

    return {
        "cohabitants_n_pris_en_compte": int(nb_debiteurs),
        "revenus_debiteurs_mensuels_total": float(revenus_debiteurs_m),
        "cohabitants_part_totale_avant_partage_mensuel": float(part_m),
        "cohabitants_part_a_compter_mensuel": float(part_m_par_dem),
        "cohabitants_part_a_compter_annuel": float(part_m_par_dem * 12.0),
    }


# ============================================================
# CALCUL GLOBAL — OFFICIEL CPAS (ANNUEL puis /12) — 1 dossier
# (PF à compter mensuel = input)
# ============================================================
def compute_officiel_cpas_annuel(answers: dict, engine: dict) -> dict:
    cfg = engine["config"]
    cat = answers.get("categorie", "isole")

    taux_ris_m = float(cfg["ris_rates"].get(cat, 0.0))
    taux_ris_annuel = taux_ris_m * 12.0

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

    # 3) Immo
    immo_ann = immo_annuel_total(
        biens=answers.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )

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

    # 5) Art.34 (version CPAS corrigée)
    art34 = cohabitants_art34_part_mensuelle_cpas(
        cohabitants=answers.get("cohabitants_art34", []),
        taux_a_laisser_mensuel=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        partage_active=bool(answers.get("partage_enfants_jeunes_actif", False)),
        nb_demandeurs_a_partager=int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
    )

    # 5bis) PF à compter (mensuel -> annuel) — ICI au niveau du demandeur (pas ménage)
    pf_m = max(0.0, float(answers.get("prestations_familiales_a_compter_mensuel", 0.0)))
    pf_ann = pf_m * 12.0

    # 6) Avantage en nature logement (mensuel -> annuel)
    avantage_nature_m = max(0.0, float(answers.get("avantage_nature_logement_mensuel", 0.0)))
    avantage_nature_ann = avantage_nature_m * 12.0

    total_avant_annuel = (
        revenus_demandeur_annuels
        + cap_ann
        + immo_ann
        + ces_ann
        + art34["cohabitants_part_a_compter_annuel"]
        + pf_ann
        + avantage_nature_ann
    )

    # Immunisation simple (annuelle) si ressources < taux
    immu_ann = 0.0
    if taux_ris_annuel > 0 and total_avant_annuel < taux_ris_annuel:
        immu_ann = float(cfg["immunisation_simple_annuelle"].get(cat, 0.0))

    total_apres_annuel = max(0.0, total_avant_annuel - immu_ann)
    ris_annuel = max(0.0, taux_ris_annuel - total_apres_annuel) if taux_ris_annuel > 0 else 0.0
    ris_mensuel = ris_annuel / 12.0

    # Prorata 1er mois
    d_dem = answers.get("date_demande", date.today())
    pr = month_prorata_from_request_date(d_dem)
    ris_premier_mois = ris_mensuel * pr["prorata"]

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

        "avantage_nature_logement_annuel": float(avantage_nature_ann),
        "total_ressources_avant_immunisation_simple_annuel": float(total_avant_annuel),
        "taux_ris_annuel": float(taux_ris_annuel),
        "immunisation_simple_annuelle": float(immu_ann),
        "total_ressources_apres_immunisation_simple_annuel": float(total_apres_annuel),
        "ris_theorique_annuel": float(ris_annuel),

        "taux_ris_mensuel": float(taux_ris_m),
        "ris_theorique_mensuel": float(ris_mensuel),

        "date_demande": str(d_dem),
        "jours_dans_mois": pr["jours_dans_mois"],
        "jours_restants_inclus": pr["jours_restants_inclus"],
        "prorata_premier_mois": pr["prorata"],
        "ris_premier_mois_prorata": float(ris_premier_mois),
        "ris_mois_suivants": float(ris_mensuel),
    }


# ============================================================
# EXPORT PDF (si reportlab dispo) + fallback texte
# ============================================================
def build_decision_text(dossier_label: str, res: dict) -> str:
    lines = []
    lines.append(f"Décision RIS — {dossier_label}")
    lines.append("")
    lines.append(f"Catégorie: {res['categorie']}")
    lines.append(f"Taux RIS mensuel: {res['taux_ris_mensuel']:.2f} €")
    lines.append("")
    lines.append("Ressources (annuel):")
    lines.append(f"- Revenus demandeur: {res['revenus_demandeur_annuels']:.2f} €")
    lines.append(f"- Capitaux mobiliers: {res['capitaux_mobiliers_annuels']:.2f} €")
    lines.append(f"- Immobilier: {res['immo_annuels']:.2f} €")
    lines.append(f"- Cession: {res['cession_biens_annuelle']:.2f} €")
    lines.append(f"- Art.34 (annuel): {res['cohabitants_part_a_compter_annuel']:.2f} €")
    lines.append(f"- Prestations familiales à compter (annuel): {res['prestations_familiales_a_compter_annuel']:.2f} €")
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
st.caption("Mode 'feuilles CPAS' : calcul ANNUEL → immunisation simple → RIS annuel → /12 + prorata 1er mois.")

engine = load_engine()
cfg = engine["config"]

with st.sidebar:
    st.subheader("Paramètres (JSON / indexables)")

    st.write("**Taux RIS mensuels**")
    cfg["ris_rates"]["cohab"] = st.number_input("RIS cohab (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["cohab"]))
    cfg["ris_rates"]["isole"] = st.number_input("RIS isolé (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["isole"]))
    cfg["ris_rates"]["fam_charge"] = st.number_input("RIS fam. charge (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["fam_charge"]))

    st.divider()
    st.write("**Art.34 : taux cat.1 à laisser (€/mois)**")
    cfg["art34"]["taux_a_laisser_mensuel"] = st.number_input(
        "Taux à laisser aux débiteurs admissibles",
        min_value=0.0,
        value=float(cfg["art34"]["taux_a_laisser_mensuel"])
    )

    st.divider()
    st.write("**Immunisation simple (€/an)**")
    cfg["immunisation_simple_annuelle"]["cohab"] = st.number_input("Immu cohab (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["cohab"]))
    cfg["immunisation_simple_annuelle"]["isole"] = st.number_input("Immu isolé (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["isole"]))
    cfg["immunisation_simple_annuelle"]["fam_charge"] = st.number_input("Immu fam. charge (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["fam_charge"]))

    st.divider()
    st.write("**Exonérations socio-pro**")
    cfg["socio_prof"]["max_mensuel"] = st.number_input("Exo socio-pro max (€/mois)", min_value=0.0, value=float(cfg["socio_prof"]["max_mensuel"]))
    cfg["socio_prof"]["artistique_annuel"] = st.number_input("Exo artistique irrégulier (€/an)", min_value=0.0, value=float(cfg["socio_prof"]["artistique_annuel"]))


# ------------------------------------------------------------
# MODE MULTI DEMANDES (hors couple) — jusqu’à 3
# ------------------------------------------------------------
st.subheader("Mode dossier")
multi_mode = st.checkbox(
    "Plusieurs demandes RIS (hors couple) — comparer jusqu’à 3 demandeurs",
    value=False
)

# --- MENAGE COMMUN (toujours encodé une fois en multi) ---
def ui_menage_commum(prefix: str) -> dict:
    answers = {}

    st.divider()
    st.subheader("Ménage (commun)")

    # Partage art34 entre plusieurs demandeurs (enfants/jeunes)
    answers["partage_enfants_jeunes_actif"] = st.checkbox(
        "Partager la part art.34 entre plusieurs ENFANTS/JEUNES demandeurs (uniquement dans ce cas)",
        value=False,
        key=f"{prefix}_partage"
    )
    answers["nb_enfants_jeunes_demandeurs"] = 1
    if answers["partage_enfants_jeunes_actif"]:
        answers["nb_enfants_jeunes_demandeurs"] = st.number_input(
            "Nombre de demandeurs à partager (ex: 2 enfants)",
            min_value=1, value=2, step=1,
            key=f"{prefix}_nb_partage"
        )

    # Cohabitants art.34
    st.markdown("### Cohabitants admissibles (art.34)")
    st.caption("Règle CPAS appliquée : somme revenus débiteurs − (nb débiteurs × taux à laisser), puis partage éventuel.")
    nb_coh = st.number_input("Nombre de cohabitants à encoder", min_value=0, value=1, step=1, key=f"{prefix}_nbcoh")
    cohabitants = []
    for i in range(int(nb_coh)):
        st.markdown(f"**Cohabitant {i+1}**")
        c1, c2 = st.columns([2, 1])
        typ = c1.selectbox("Type", ["partenaire", "debiteur_direct_1", "debiteur_direct_2", "autre"], key=f"{prefix}_coh_t_{i}")
        rev = c2.number_input("Revenus nets annuels (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_coh_r_{i}")
        excl = st.checkbox("Ne pas prendre en compte (équité / décision CPAS)", value=False, key=f"{prefix}_coh_x_{i}")
        cohabitants.append({"type": typ, "revenu_net_annuel": float(rev), "exclure": bool(excl)})
    answers["cohabitants_art34"] = cohabitants

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


def ui_revenus_annuels_block(prefix: str) -> list:
    lst = []
    nb = st.number_input(f"Nombre de revenus à encoder ({prefix})", min_value=0, value=1, step=1, key=f"{prefix}_nb")
    for i in range(int(nb)):
        st.markdown(f"**Revenu {i+1} ({prefix})**")
        c1, c2, c3 = st.columns([2, 1, 1])
        label = c1.text_input("Type/label", value="salaire/chômage", key=f"{prefix}_lab_{i}")
        montant_a = c2.number_input("Montant net ANNUEL (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_a_{i}")
        typ = c3.selectbox("Règle", ["standard", "socio_prof", "etudiant", "artistique_irregulier", "ale"], key=f"{prefix}_t_{i}")

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


# ------------------------------------------------------------
# MODE MULTI
# ------------------------------------------------------------
if multi_mode:
    st.subheader("A) Demandeurs (jusque 3)")
    nb_dem = st.number_input("Nombre de demandeurs à comparer", min_value=2, max_value=3, value=2, step=1)

    demandeurs = []
    for i in range(int(nb_dem)):
        st.markdown(f"### Demandeur {i+1}")
        label = st.text_input("Nom/Label", value=f"Demandeur {i+1}", key=f"md_lab_{i}")
        cat = st.selectbox("Catégorie RIS", ["cohab", "isole", "fam_charge"], key=f"md_cat_{i}")
        enfants = st.number_input("Enfants à charge", min_value=0, value=0, step=1, key=f"md_enf_{i}")
        d_dem = st.date_input("Date de demande", value=date.today(), key=f"md_date_{i}")

        st.markdown("**Revenus nets ANNUELS (demandeur)**")
        revs = ui_revenus_annuels_block(f"md_rev_{i}")

        st.markdown("**Prestations familiales à compter (spécifiques à CE demandeur)**")
        pf_m = st.number_input(
            "PF à compter (€/mois) (ex: 240 pour A, 0 pour B)",
            min_value=0.0, value=0.0, step=10.0,
            key=f"md_pf_{i}"
        )

        demandeurs.append({
            "label": label,
            "categorie": cat,
            "enfants_a_charge": int(enfants),
            "date_demande": d_dem,
            "revenus_demandeur_annuels": revs,
            "prestations_familiales_a_compter_mensuel": float(pf_m),
        })

    menage = ui_menage_commum("md_menage")

    st.divider()
    if st.button("Calculer (comparatif)"):
        results = []
        for d in demandeurs:
            # On fabrique un answers "complet" par demandeur en injectant les champs ménage
            answers = dict(menage)
            answers.update({
                "categorie": d["categorie"],
                "enfants_a_charge": d["enfants_a_charge"],
                "date_demande": d["date_demande"],
                "couple_demandeur": False,  # multi hors couple
                "revenus_demandeur_annuels": d["revenus_demandeur_annuels"],
                "revenus_conjoint_annuels": [],
                "prestations_familiales_a_compter_mensuel": d["prestations_familiales_a_compter_mensuel"],
            })

            # IMPORTANT: si tu coches le partage, par défaut on partage entre nb_dem demandeurs
            if bool(answers.get("partage_enfants_jeunes_actif", False)):
                # si l’utilisateur a laissé un nombre “à partager”, on le garde.
                # sinon, on force à nb_dem.
                if int(answers.get("nb_enfants_jeunes_demandeurs", 1)) <= 1:
                    answers["nb_enfants_jeunes_demandeurs"] = int(nb_dem)

            res = compute_officiel_cpas_annuel(answers, engine)
            res["_label"] = d["label"]
            results.append(res)

        st.success("Calcul terminé ✅")

        st.markdown("## Tableau comparatif")
        table_rows = []
        for r in results:
            table_rows.append({
                "Demandeur": r["_label"],
                "Catégorie": r["categorie"],
                "RIS mensuel": round(r["ris_theorique_mensuel"], 2),
                "RIS 1er mois (prorata)": round(r["ris_premier_mois_prorata"], 2),
                "Art.34 mensuel compté": round(r["cohabitants_part_a_compter_mensuel"], 2),
                "PF mensuel compté": round(r["prestations_familiales_a_compter_mensuel"], 2),
                "Total ressources (annuel)": round(r["total_ressources_avant_immunisation_simple_annuel"], 2),
            })
        st.dataframe(table_rows, use_container_width=True)

        st.divider()
        st.markdown("## Détails (par demandeur)")
        for r in results:
            with st.expander(f"Détail — {r['_label']}"):
                st.metric("RIS mensuel", f"{r['ris_theorique_mensuel']:.2f} €")
                st.metric("RIS 1er mois (prorata)", f"{r['ris_premier_mois_prorata']:.2f} €")
                st.json(r)

                # Exports
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
                    st.info(
                        "PDF indisponible ici (reportlab non installé). "
                        "Sur Streamlit Cloud: ajoute `reportlab` dans requirements.txt."
                    )

else:
    # ------------------------------------------------------------
    # MODE 1 DEMANDE (ton ancien flux, en gardant l’esprit)
    # ------------------------------------------------------------
    answers = {}

    st.subheader("Profil")
    answers["categorie"] = st.selectbox("Catégorie RIS", ["cohab", "isole", "fam_charge"])
    answers["enfants_a_charge"] = st.number_input("Enfants à charge", min_value=0, value=0, step=1)

    st.divider()
    st.subheader("Date de la demande (pour calcul du 1er mois)")
    answers["date_demande"] = st.date_input("Date de la demande", value=date.today())

    st.divider()
    st.subheader("1) Revenus du demandeur — ANNUELS (nets)")
    answers["couple_demandeur"] = st.checkbox(
        "Demande introduite par un COUPLE (2 demandeurs ensemble)",
        value=False
    )

    st.markdown("**Demandeur 1**")
    answers["revenus_demandeur_annuels"] = ui_revenus_annuels_block("dem")

    answers["revenus_conjoint_annuels"] = []
    if answers["couple_demandeur"]:
        st.divider()
        st.markdown("**Demandeur 2 (conjoint/partenaire) — revenus à additionner**")
        answers["revenus_conjoint_annuels"] = ui_revenus_annuels_block("conj")

    st.divider()
    st.subheader("PF à compter (spécifiques au demandeur)")
    answers["prestations_familiales_a_compter_mensuel"] = st.number_input(
        "Prestations familiales à compter (€/mois)",
        min_value=0.0, value=0.0, step=10.0
    )

    # Ménage (reprend ton encoding)
    st.divider()
    st.subheader("Ménage")

    answers["partage_enfants_jeunes_actif"] = st.checkbox(
        "Partager la part art.34 entre plusieurs ENFANTS/JEUNES demandeurs (uniquement dans ce cas)",
        value=False
    )
    answers["nb_enfants_jeunes_demandeurs"] = 1
    if answers["partage_enfants_jeunes_actif"]:
        answers["nb_enfants_jeunes_demandeurs"] = st.number_input(
            "Nombre d'enfants/jeunes demandeurs à partager",
            min_value=1, value=2, step=1
        )

    st.markdown("### Cohabitants (art.34)")
    cohabitants = []
    nb_coh = st.number_input("Nombre de cohabitants à encoder", min_value=0, value=1, step=1)
    for i in range(int(nb_coh)):
        st.markdown(f"**Cohabitant {i+1}**")
        c1, c2 = st.columns([2, 1])
        typ = c1.selectbox("Type", ["partenaire", "debiteur_direct_1", "debiteur_direct_2", "autre"], key=f"coh_type_{i}")
        rev = c2.number_input("Revenus nets annuels (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"coh_rev_{i}")
        excl = st.checkbox("Ne pas prendre en compte (équité / décision CPAS)", value=False, key=f"coh_excl_{i}")
        cohabitants.append({"type": typ, "revenu_net_annuel": float(rev), "exclure": bool(excl)})
    answers["cohabitants_art34"] = cohabitants

    st.divider()
    st.subheader("2) Capitaux mobiliers")
    a_cap = st.checkbox("Le(s) demandeur(s) possède(nt) des capitaux mobiliers")
    answers["capital_mobilier_total"] = 0.0
    answers["capital_compte_commun"] = False
    answers["capital_nb_titulaires"] = 1
    answers["capital_conjoint_cotitulaire"] = False
    answers["capital_fraction"] = 1.0
    if a_cap:
        answers["capital_mobilier_total"] = st.number_input("Montant total capitaux (€)", min_value=0.0, value=0.0, step=100.0)
        answers["capital_compte_commun"] = st.checkbox("Compte commun ?", value=False)
        if answers["capital_compte_commun"]:
            answers["capital_nb_titulaires"] = st.number_input("Nombre de titulaires", min_value=1, value=2, step=1)
            if answers["categorie"] == "fam_charge":
                answers["capital_conjoint_cotitulaire"] = st.checkbox("Le conjoint/partenaire est co-titulaire ?", value=True)
        else:
            answers["capital_fraction"] = st.number_input("Part du ménage demandeur (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

    st.divider()
    st.subheader("3) Biens immobiliers")
    biens = []
    a_immo = st.checkbox("Le ménage demandeur possède des biens immobiliers")
    if a_immo:
        nb_biens = st.number_input("Nombre de biens", min_value=0, value=1, step=1)
        for i in range(int(nb_biens)):
            st.markdown(f"**Bien {i+1}**")
            habitation_principale = st.checkbox("Habitation principale ?", value=False, key=f"im_hp_{i}")
            bati = st.checkbox("Bien bâti ?", value=True, key=f"im_bati_{i}")
            rc_non_indexe = st.number_input("RC non indexé annuel", min_value=0.0, value=0.0, step=50.0, key=f"im_rc_{i}")
            fraction = st.number_input("Fraction de droits (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"im_frac_{i}")

            hypotheque = False
            interets_annuels = 0.0
            viager = False
            rente = 0.0

            if not habitation_principale:
                hypotheque = st.checkbox("Hypothèque ?", value=False, key=f"im_hyp_{i}")
                if hypotheque:
                    interets_annuels = st.number_input("Intérêts hypothécaires annuels", min_value=0.0, value=0.0, step=50.0, key=f"im_int_{i}")

                viager = st.checkbox("Acquis en viager ?", value=False, key=f"im_vi_{i}")
                if viager:
                    rente = st.number_input("Rente viagère annuelle", min_value=0.0, value=0.0, step=50.0, key=f"im_rente_{i}")

            biens.append({
                "habitation_principale": habitation_principale,
                "bati": bati,
                "rc_non_indexe": float(rc_non_indexe),
                "fraction_droits": float(fraction),
                "hypotheque": hypotheque,
                "interets_annuels": float(interets_annuels),
                "viager": viager,
                "rente_viagere_annuelle": float(rente)
            })
    answers["biens_immobiliers"] = biens

    st.divider()
    st.subheader("4) Cession de biens")
    cessions = []
    a_ces = st.checkbox("Le ménage demandeur a cédé des biens (10 dernières années)")
    answers["cessions"] = []
    answers["cession_cas_particulier_37200"] = False
    answers["cession_dettes_deductibles"] = 0.0
    answers["cession_abatt_cat"] = "cat1"
    answers["cession_abatt_mois"] = 0
    if a_ces:
        answers["cession_cas_particulier_37200"] = st.checkbox("Cas particulier: tranche immunisée 37.200€", value=False)
        dettes_ok = st.checkbox("Déduire des dettes personnelles ?", value=False)
        if dettes_ok:
            answers["cession_dettes_deductibles"] = st.number_input("Dettes déductibles (€)", min_value=0.0, value=0.0, step=100.0)
        answers["cession_abatt_cat"] = st.selectbox("Catégorie d’abattement", ["cat1", "cat2", "cat3"])
        answers["cession_abatt_mois"] = st.number_input("Prorata mois", min_value=0, max_value=12, value=0, step=1)
        nb_c = st.number_input("Nombre de cessions", min_value=0, value=1, step=1)
        for i in range(int(nb_c)):
            st.markdown(f"**Cession {i+1}**")
            val = st.number_input("Valeur vénale (€)", min_value=0.0, value=0.0, step=100.0, key=f"ces_v_{i}")
            usuf = st.checkbox("Usufruit ?", value=False, key=f"ces_u_{i}")
            cessions.append({"valeur_venale": float(val), "usufruit": bool(usuf)})
        answers["cessions"] = cessions

    st.divider()
    st.subheader("6) Avantage en nature")
    answers["avantage_nature_logement_mensuel"] = st.number_input(
        "Logement payé par un tiers non cohabitant (€/mois) — montant à compter",
        min_value=0.0, value=0.0, step=10.0
    )

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
