import json
import os
import calendar
from datetime import date

import streamlit as st


# ============================================================
# CONFIG PAR DÉFAUT (fusion avec ris_rules.json si présent)
# ============================================================
DEFAULT_ENGINE = {
    "version": "1.0",
    "config": {
        "ris_rates": {"cohab": 876.13, "isole": 1314.20, "fam_charge": 1776.07},
        "immunisation_simple_annuelle": {"cohab": 155.0, "isole": 250.0, "fam_charge": 310.0},
        "art34": {"taux_a_laisser_mensuel": 876.13},
        "capital_mobilier": {
            "t0_max": 6199.0,
            "t1_min": 6200.0,
            "t1_max": 12500.0,
            "t1_rate": 0.06,
            "t2_rate": 0.10
        },
        "immo": {"bati_base": 750.0, "bati_par_enfant": 125.0, "non_bati_base": 30.0, "coeff_rc": 3.0},
        "socio_prof": {"max_mensuel": 274.82, "artistique_annuel": 3297.80},
        "cession": {
            "tranche_immunisee": 37200.0,
            "usufruit_ratio": 0.40,
            "abattements_annuels": {"cat1": 1250.0, "cat2": 2000.0, "cat3": 2500.0}
        },
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
# REVENUS (on encode en ANNUEL, exo en mensuel, retour en annuel)
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
# CAPITAUX MOBILIERS (annuel)
# ============================================================
def capital_mobilier_annuel(total_capital: float, cfg_cap: dict) -> float:
    total = max(0.0, float(total_capital))
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
# IMMOBILIER (annuel) — version simplifiée RC*3 - exonérations
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
        total_annuel += max(0.0, base)

    return float(max(0.0, total_annuel))


# ============================================================
# CESSION (annuel) — version simplifiée
# ============================================================
def cession_biens_annuelle(cessions: list, cfg_cession: dict, cfg_cap: dict) -> float:
    total = 0.0
    for c in cessions:
        v = max(0.0, float(c.get("valeur_venale", 0.0)))
        if c.get("usufruit", False):
            v *= float(cfg_cession["usufruit_ratio"])
        total += v

    # barème comme capitaux
    return capital_mobilier_annuel(total, cfg_cap)


# ============================================================
# ART.34 — cohabitants admissibles
# Feuille CPAS : part = max(0, revenu_mensuel - taux_cat1_a_laisser)
# ============================================================
ADMISSIBLES_ART34 = {"partenaire", "debiteur_direct_1", "debiteur_direct_2"}


def cohabitants_art34_part_annuelle(
    cohabitants: list,
    taux_a_laisser_mensuel: float,
    partage_active: bool,
    nb_enfants_jeunes_demandeurs: int
) -> dict:
    taux = max(0.0, float(taux_a_laisser_mensuel))
    total_part_m = 0.0
    n_pris = 0

    for c in cohabitants:
        typ = c.get("type", "autre")
        if typ not in ADMISSIBLES_ART34:
            continue
        if bool(c.get("exclure", False)):
            continue

        revenu_ann = max(0.0, float(c.get("revenu_net_annuel", 0.0)))
        revenu_m = revenu_ann / 12.0
        part_m = max(0.0, revenu_m - taux)

        total_part_m += part_m
        n_pris += 1

    if partage_active:
        n = max(1, int(nb_enfants_jeunes_demandeurs))
        total_part_m_partagee = total_part_m / n
    else:
        total_part_m_partagee = total_part_m

    return {
        "cohabitants_n_pris_en_compte": int(n_pris),
        "cohabitants_part_totale_avant_partage_mensuel": float(total_part_m),
        "cohabitants_part_a_compter_mensuel": float(total_part_m_partagee),
        "cohabitants_part_a_compter_annuel": float(total_part_m_partagee * 12.0),
    }


# ============================================================
# CALCUL OFFICIEL CPAS (ANNUEL puis /12) — pour 1 demandeur
# ============================================================
def compute_officiel_cpas_annuel(dem: dict, menage: dict, engine: dict) -> dict:
    cfg = engine["config"]

    cat = dem["categorie"]
    taux_ris_m = float(cfg["ris_rates"].get(cat, 0.0))
    taux_ris_annuel = taux_ris_m * 12.0

    revenus_demandeur_annuels = revenus_annuels_apres_exonerations(dem["revenus_annuels"], cfg["socio_prof"])

    cap_ann = capital_mobilier_annuel(menage.get("capital_mobilier_total", 0.0), cfg["capital_mobilier"])
    immo_ann = immo_annuel_total(menage.get("biens_immobiliers", []), dem.get("enfants_a_charge", 0), cfg["immo"])
    ces_ann = cession_biens_annuelle(menage.get("cessions", []), cfg["cession"], cfg["capital_mobilier"])

    art34 = cohabitants_art34_part_annuelle(
        cohabitants=menage.get("cohabitants_art34", []),
        taux_a_laisser_mensuel=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        partage_active=bool(menage.get("partage_enfants_jeunes_actif", False)),
        nb_enfants_jeunes_demandeurs=int(menage.get("nb_enfants_jeunes_demandeurs", 1)),
    )

    avantage_nature_m = max(0.0, float(menage.get("avantage_nature_logement_mensuel", 0.0)))
    avantage_nature_ann = avantage_nature_m * 12.0

    total_avant_annuel = (
        revenus_demandeur_annuels
        + cap_ann
        + immo_ann
        + ces_ann
        + art34["cohabitants_part_a_compter_annuel"]
        + avantage_nature_ann
    )

    immu_ann = 0.0
    if taux_ris_annuel > 0 and total_avant_annuel < taux_ris_annuel:
        immu_ann = float(cfg["immunisation_simple_annuelle"].get(cat, 0.0))

    total_apres_annuel = max(0.0, total_avant_annuel - immu_ann)
    ris_annuel = max(0.0, taux_ris_annuel - total_apres_annuel) if taux_ris_annuel > 0 else 0.0
    ris_mensuel = ris_annuel / 12.0

    pr = month_prorata_from_request_date(dem["date_demande"])
    ris_premier_mois = ris_mensuel * pr["prorata"]

    return {
        "mode_calcul": "OFFICIEL_CPAS_ANNUEL_MULTI",
        "demandeur": dem["nom"],
        "categorie": cat,
        "enfants_a_charge": int(dem.get("enfants_a_charge", 0)),

        "partage_enfants_jeunes_actif": bool(menage.get("partage_enfants_jeunes_actif", False)),
        "nb_enfants_jeunes_demandeurs": int(menage.get("nb_enfants_jeunes_demandeurs", 1)),

        "revenus_demandeur_annuels": float(revenus_demandeur_annuels),
        "capitaux_mobiliers_annuels": float(cap_ann),
        "immo_annuels": float(immo_ann),
        "cession_biens_annuelle": float(ces_ann),
        **art34,

        "avantage_nature_logement_annuel": float(avantage_nature_ann),
        "total_ressources_avant_immunisation_simple_annuel": float(total_avant_annuel),
        "taux_ris_annuel": float(taux_ris_annuel),
        "immunisation_simple_annuelle": float(immu_ann),
        "total_ressources_apres_immunisation_simple_annuel": float(total_apres_annuel),

        "taux_ris_mensuel": float(taux_ris_m),
        "ris_theorique_mensuel": float(ris_mensuel),

        "date_demande": str(dem["date_demande"]),
        "jours_dans_mois": pr["jours_dans_mois"],
        "jours_restants_inclus": pr["jours_restants_inclus"],
        "prorata_premier_mois": pr["prorata"],
        "ris_premier_mois_prorata": float(ris_premier_mois),
        "ris_mois_suivants": float(ris_mensuel),
    }


# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(page_title="Calcul RIS (CPAS officiel) — multi", layout="centered")

if os.path.exists("logo.png"):
    st.image("logo.png", use_container_width=False)

st.title("Calcul RIS — Prototype (CPAS officiel : multi-demandeurs)")
st.caption("Encode le ménage une fois → choisis le demandeur → résultat RIS (annuel CPAS puis /12 + prorata 1er mois).")

engine = load_engine()
cfg = engine["config"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Paramètres")
    cfg["art34"]["taux_a_laisser_mensuel"] = st.number_input(
        "Art.34 — taux cat.1 à laisser (€/mois)",
        min_value=0.0,
        value=float(cfg["art34"]["taux_a_laisser_mensuel"])
    )

# ---------------- Demandeurs ----------------
st.subheader("A) Demandeurs (plusieurs)")
nb_dem = st.number_input("Nombre de demandeurs à encoder", min_value=1, value=2, step=1)

demandeurs = []
for i in range(int(nb_dem)):
    st.markdown(f"### Demandeur {i+1}")
    nom = st.text_input("Nom / label", value=f"Demandeur {i+1}", key=f"dem_nom_{i}")
    cat = st.selectbox("Catégorie RIS", ["cohab", "isole", "fam_charge"], key=f"dem_cat_{i}")
    enfants = st.number_input("Enfants à charge", min_value=0, value=0, step=1, key=f"dem_enf_{i}")
    d_dem = st.date_input("Date de demande", value=date.today(), key=f"dem_date_{i}")

    # Revenus annuels
    st.markdown("**Revenus nets ANNUELS (demandeur)**")
    nb_rev = st.number_input("Nombre de revenus", min_value=0, value=1, step=1, key=f"dem_nbrev_{i}")
    revs = []
    for j in range(int(nb_rev)):
        c1, c2, c3 = st.columns([2, 1, 1])
        label = c1.text_input("Label", value="salaire/chômage", key=f"dem_lab_{i}_{j}")
        montant = c2.number_input("Montant net annuel (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"dem_m_{i}_{j}")
        typ = c3.selectbox("Règle", ["standard", "socio_prof", "etudiant", "artistique_irregulier", "ale"], key=f"dem_t_{i}_{j}")
        eligible = True
        ale_part_exc_m = 0.0
        if typ in ("socio_prof", "etudiant", "artistique_irregulier"):
            eligible = st.checkbox("Éligible exonération ?", value=True, key=f"dem_el_{i}_{j}")
        if typ == "ale":
            ale_part_exc_m = st.number_input("Part ALE à compter (>4,10€) (€/mois)", min_value=0.0, value=0.0, step=1.0, key=f"dem_ale_{i}_{j}")

        revs.append({
            "label": label,
            "montant_annuel": float(montant),
            "type": typ,
            "eligible": eligible,
            "ale_part_excedentaire_mensuel": float(ale_part_exc_m),
        })

    demandeurs.append({
        "nom": nom,
        "categorie": cat,
        "enfants_a_charge": int(enfants),
        "date_demande": d_dem,
        "revenus_annuels": revs
    })

st.divider()

# ---------------- MENAGE (commun) ----------------
st.subheader("B) Ménage (commun à tous les demandeurs)")

menage = {}

# Partage enfants/jeunes (uniquement si plusieurs jeunes demandeurs)
menage["partage_enfants_jeunes_actif"] = st.checkbox(
    "Partager la part art.34 entre plusieurs ENFANTS/JEUNES demandeurs (uniquement dans ce cas)",
    value=False
)
menage["nb_enfants_jeunes_demandeurs"] = 1
if menage["partage_enfants_jeunes_actif"]:
    menage["nb_enfants_jeunes_demandeurs"] = st.number_input(
        "Nombre d'enfants/jeunes demandeurs à partager",
        min_value=1, value=2, step=1
    )

# Cohabitants art.34
st.markdown("### Cohabitants (art.34)")
nb_coh = st.number_input("Nombre de cohabitants à encoder", min_value=0, value=2, step=1)
cohabitants = []
for i in range(int(nb_coh)):
    st.markdown(f"**Cohabitant {i+1}**")
    c1, c2 = st.columns([2, 1])
    typ = c1.selectbox("Type", ["partenaire", "debiteur_direct_1", "debiteur_direct_2", "autre"], key=f"coh_t_{i}")
    rev = c2.number_input("Revenus nets annuels (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"coh_r_{i}")
    excl = st.checkbox("Ne pas prendre en compte (équité / décision CPAS)", value=False, key=f"coh_x_{i}")

    cohabitants.append({"type": typ, "revenu_net_annuel": float(rev), "exclure": bool(excl)})

menage["cohabitants_art34"] = cohabitants

# Capitaux
st.markdown("### Capitaux mobiliers (ménage)")
menage["capital_mobilier_total"] = st.number_input("Capitaux totaux (€)", min_value=0.0, value=0.0, step=100.0)

# Immo
st.markdown("### Biens immobiliers (ménage)")
biens = []
a_immo = st.checkbox("Le ménage possède des biens immobiliers", value=False)
if a_immo:
    nb_biens = st.number_input("Nombre de biens", min_value=0, value=1, step=1)
    for i in range(int(nb_biens)):
        st.markdown(f"**Bien {i+1}**")
        habitation_principale = st.checkbox("Habitation principale ?", value=False, key=f"im_hp_{i}")
        bati = st.checkbox("Bien bâti ?", value=True, key=f"im_b_{i}")
        rc = st.number_input("RC non indexé annuel", min_value=0.0, value=0.0, step=50.0, key=f"im_rc_{i}")
        frac = st.number_input("Fraction droits (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"im_f_{i}")
        biens.append({"habitation_principale": habitation_principale, "bati": bati, "rc_non_indexe": float(rc), "fraction_droits": float(frac)})
menage["biens_immobiliers"] = biens

# Cession
st.markdown("### Cession de biens (ménage)")
cessions = []
a_ces = st.checkbox("Le ménage a cédé des biens", value=False)
if a_ces:
    nb_c = st.number_input("Nombre de cessions", min_value=0, value=1, step=1)
    for i in range(int(nb_c)):
        st.markdown(f"**Cession {i+1}**")
        val = st.number_input("Valeur vénale (€)", min_value=0.0, value=0.0, step=100.0, key=f"ces_v_{i}")
        usuf = st.checkbox("Usufruit ?", value=False, key=f"ces_u_{i}")
        cessions.append({"valeur_venale": float(val), "usufruit": bool(usuf)})
menage["cessions"] = cessions

# Avantage nature
st.markdown("### Avantage en nature")
menage["avantage_nature_logement_mensuel"] = st.number_input("Logement payé par un tiers (€/mois)", min_value=0.0, value=0.0, step=10.0)

st.divider()

# ---------------- CALCUL ----------------
st.subheader("C) Calcul")
choix = st.selectbox("Calculer le RIS pour :", [d["nom"] for d in demandeurs])

if st.button("Calculer"):
    dem = next(d for d in demandeurs if d["nom"] == choix)
    res = compute_officiel_cpas_annuel(dem, menage, engine)

    st.success("Calcul terminé ✅")
    st.metric("RIS mensuel normal (€/mois)", f"{res['ris_theorique_mensuel']:.2f}")
    st.metric("RIS du 1er mois (prorata)", f"{res['ris_premier_mois_prorata']:.2f}")

    st.caption(
        f"Prorata = {res['jours_restants_inclus']}/{res['jours_dans_mois']} "
        f"= {res['prorata_premier_mois']:.6f} (jour inclus)"
    )

    st.write("### Détail")
    st.json(res)
