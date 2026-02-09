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
        # Taux RIS (mensuel)
        "ris_rates": {"cohab": 876.13, "isole": 1314.20, "fam_charge": 1776.07},

        # Immunisation simple (annuelle)
        "immunisation_simple_annuelle": {"cohab": 155.0, "isole": 250.0, "fam_charge": 310.0},

        # Art. 34 : taux "catégorie 1 à laisser" (mensuel) pour cohabitants admissibles
        # Dans tes feuilles: 876,13€ (souvent = RIS cohabitant)
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

    # Types
    for k in ("cohab", "isole", "fam_charge"):
        cfg["ris_rates"][k] = float(cfg["ris_rates"].get(k, 0.0))
        cfg["immunisation_simple_annuelle"][k] = float(cfg["immunisation_simple_annuelle"].get(k, 0.0))

    # Art34 default: = RIS cohab si absent
    if "art34" not in cfg:
        cfg["art34"] = {}
    cfg["art34"]["taux_a_laisser_mensuel"] = float(cfg["art34"].get("taux_a_laisser_mensuel", cfg["ris_rates"]["cohab"]))

    return engine


def load_engine() -> dict:
    if os.path.exists("ris_rules.json"):
        with open("ris_rules.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        return normalize_engine(raw)
    return normalize_engine(DEFAULT_ENGINE)


def month_prorata_from_request_date(d: date) -> dict:
    """
    Prorata du mois en cours, jour de demande inclus.
    jours_restants_inclus = nb_jours_mois - jour + 1
    """
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

    annuel = 0.0
    if adj_total <= t0_max:
        return 0.0

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

        # Hypothèque: intérêts plafonnés à 50% du montant compté
        if b.get("hypotheque", False):
            interets = max(0.0, float(b.get("interets_annuels", 0.0))) * frac
            base -= min(interets, 0.5 * base)

        # Viager: rente plafonnée à 50%
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

    # Tranches capitaux (non fractionnées ici)
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
# REVENUS (exo socio-pro) -> ici on encode en ANNUEL dans l'UI
# On convertit en MENSUEL pour appliquer règles mensuelles (exo), puis on remonte en ANNUEL.
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
# ART. 34 — cohabitants admissibles (ANNUEL)
# Règle des feuilles:
#   part_mensuelle = max(0, revenu_mensuel - taux_cat1_a_laisser)
#   part_annuelle = part_mensuelle * 12
# Puis (optionnel) partage UNIQUEMENT si plusieurs enfants/jeunes demandeurs.
# ============================================================
ADMISSIBLES_ART34 = {
    "partenaire",          # partenaire/conjoint si NON-demandeur
    "debiteur_direct_1",   # parent/enfant (1er degré) (dans les deux sens)
    "debiteur_direct_2",   # grand-parent/petit-enfant (2e degré) (dans les deux sens)
}

def cohabitants_art34_part_annuelle(cohabitants: list,
                                    taux_a_laisser_mensuel: float,
                                    partage_active: bool,
                                    nb_enfants_jeunes_demandeurs: int) -> dict:
    taux = max(0.0, float(taux_a_laisser_mensuel))
    total_part_m = 0.0
    n_pris = 0

    prestations_fam_m = 0.0  # option si tu veux compter PF “en faveur du demandeur” (souvent exonéré)
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

        if bool(c.get("pf_pour_demandeur", False)):
            # On laisse l'utilisateur décider via un champ séparé (souvent = 0 car exonéré art.22)
            prestations_fam_m += max(0.0, float(c.get("pf_mensuel_a_compter", 0.0)))

    # Partage: UNIQUEMENT si activé explicitement (ex: plusieurs enfants/jeunes demandeurs)
    if partage_active:
        n = max(1, int(nb_enfants_jeunes_demandeurs))
        total_part_m_partagee = total_part_m / n
    else:
        total_part_m_partagee = total_part_m

    return {
        "cohabitants_n_pris_en_compte": int(n_pris),
        "cohabitants_part_totale_avant_partage_mensuel": float(total_part_m),
        "cohabitants_part_a_compter_mensuel": float(total_part_m_partagee),
        "prestations_familiales_a_compter_mensuel": float(prestations_fam_m),
        "cohabitants_part_totale_avant_partage_annuel": float(total_part_m * 12.0),
        "cohabitants_part_a_compter_annuel": float(total_part_m_partagee * 12.0),
        "prestations_familiales_a_compter_annuel": float(prestations_fam_m * 12.0),
    }


# ============================================================
# CALCUL GLOBAL — OFFICIEL CPAS (ANNUEL puis /12)
# ============================================================
def compute_officiel_cpas_annuel(answers: dict, engine: dict) -> dict:
    cfg = engine["config"]

    cat = answers.get("categorie", "isole")
    taux_ris_m = float(cfg["ris_rates"].get(cat, 0.0))
    taux_ris_annuel = taux_ris_m * 12.0

    # 1) Revenus demandeur (et conjoint si couple demandeur)
    revenus_demandeur_annuels = revenus_annuels_apres_exonerations(
        answers.get("revenus_demandeur_annuels", []),
        cfg["socio_prof"]
    )

    # Couple demandeur : on additionne les revenus du conjoint dans les "revenus demandeur"
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

    # 5) Art.34 cohabitants admissibles
    art34 = cohabitants_art34_part_annuelle(
        cohabitants=answers.get("cohabitants_art34", []),
        taux_a_laisser_mensuel=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        partage_active=bool(answers.get("partage_enfants_jeunes_actif", False)),
        nb_enfants_jeunes_demandeurs=int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
    )

    # 6) Avantage en nature logement (mensuel -> annuel)
    avantage_nature_m = max(0.0, float(answers.get("avantage_nature_logement_mensuel", 0.0)))
    avantage_nature_ann = avantage_nature_m * 12.0

    total_avant_annuel = (
        revenus_demandeur_annuels
        + cap_ann
        + immo_ann
        + ces_ann
        + art34["cohabitants_part_a_compter_annuel"]
        + art34["prestations_familiales_a_compter_annuel"]
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

    # Détail retour
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
# UI STREAMLIT
# ============================================================
st.set_page_config(page_title="Calcul RIS (CPAS officiel)", layout="centered")

# Logo (affiché si logo.png est dans le repo)
if os.path.exists("logo.png"):
    st.image("logo.png", use_container_width=False)

st.title("Calcul RIS — Prototype (CPAS officiel : annuel puis /12)")
st.caption("Mode 'feuilles CPAS' : calcul ANNUEL → immunisation simple → RIS annuel → /12 + prorata 1er mois.")

engine = load_engine()
cfg = engine["config"]

# ---------------- Sidebar config ----------------
with st.sidebar:
    st.subheader("Paramètres (JSON / indexables)")

    st.write("**Taux RIS mensuels (officiels)**")
    cfg["ris_rates"]["cohab"] = st.number_input("RIS cohab (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["cohab"]))
    cfg["ris_rates"]["isole"] = st.number_input("RIS isolé (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["isole"]))
    cfg["ris_rates"]["fam_charge"] = st.number_input("RIS fam. charge (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["fam_charge"]))

    st.divider()
    st.write("**Art.34 : taux cat.1 à laisser (€/mois)**")
    cfg["art34"]["taux_a_laisser_mensuel"] = st.number_input(
        "Taux à laisser aux cohabitants admissibles",
        min_value=0.0,
        value=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        help="Dans tes exemples : 876,13 (souvent = RIS cohab)."
    )

    st.divider()
    st.write("**Immunisation simple (€/an)**")
    cfg["immunisation_simple_annuelle"]["cohab"] = st.number_input("Immu simple cohab (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["cohab"]))
    cfg["immunisation_simple_annuelle"]["isole"] = st.number_input("Immu simple isolé (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["isole"]))
    cfg["immunisation_simple_annuelle"]["fam_charge"] = st.number_input("Immu simple fam. charge (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["fam_charge"]))

    st.divider()
    st.write("**Exonérations socio-pro**")
    cfg["socio_prof"]["max_mensuel"] = st.number_input("Exo socio-pro max (€/mois)", min_value=0.0, value=float(cfg["socio_prof"]["max_mensuel"]))
    cfg["socio_prof"]["artistique_annuel"] = st.number_input("Exo artistique irrégulier (€/an)", min_value=0.0, value=float(cfg["socio_prof"]["artistique_annuel"]))

# ---------------- Answers ----------------
answers = {}

st.subheader("Profil")
answers["categorie"] = st.selectbox("Catégorie RIS", ["cohab", "isole", "fam_charge"])
answers["enfants_a_charge"] = st.number_input("Enfants à charge", min_value=0, value=0, step=1)

st.divider()
st.subheader("Date de la demande (pour calcul du 1er mois)")
answers["date_demande"] = st.date_input("Date de la demande", value=date.today())

# ---------------- Demandeur ----------------
st.divider()
st.subheader("1) Revenus du demandeur — ANNUELS (nets)")

answers["couple_demandeur"] = st.checkbox(
    "Demande introduite par un COUPLE (2 demandeurs ensemble)",
    value=False,
    help="Si oui : on additionne les revenus des 2 demandeurs dans cette section."
)

def ui_revenus_annuels_block(prefix: str) -> list:
    lst = []
    nb = st.number_input(f"Nombre de revenus à encoder ({prefix})", min_value=0, value=1, step=1, key=f"{prefix}_nb")
    for i in range(int(nb)):
        st.markdown(f"**Revenu {i+1} ({prefix})**")
        c1, c2, c3 = st.columns([2, 1, 1])
        label = c1.text_input(f"Type/label", value="salaire/chômage", key=f"{prefix}_lab_{i}")
        montant_a = c2.number_input(f"Montant net ANNUEL (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_a_{i}")
        typ = c3.selectbox(f"Règle", ["standard", "socio_prof", "etudiant", "artistique_irregulier", "ale"], key=f"{prefix}_t_{i}")

        eligible = True
        ale_part_exc_m = 0.0
        if typ in ("socio_prof", "etudiant", "artistique_irregulier"):
            eligible = st.checkbox(f"Éligible exonération ?", value=True, key=f"{prefix}_el_{i}")
        if typ == "ale":
            ale_part_exc_m = st.number_input(f"Part ALE à compter (>4,10€) (€/mois)", min_value=0.0, value=0.0, step=1.0, key=f"{prefix}_ale_{i}")

        lst.append({
            "label": label,
            "montant_annuel": float(montant_a),
            "type": typ,
            "eligible": eligible,
            "ale_part_excedentaire_mensuel": float(ale_part_exc_m)
        })
    return lst

st.markdown("**Demandeur 1**")
answers["revenus_demandeur_annuels"] = ui_revenus_annuels_block("dem")

answers["revenus_conjoint_annuels"] = []
if answers["couple_demandeur"]:
    st.divider()
    st.markdown("**Demandeur 2 (conjoint/partenaire) — revenus à additionner**")
    answers["revenus_conjoint_annuels"] = ui_revenus_annuels_block("conj")

# ---------------- Capitaux ----------------
st.divider()
st.subheader("2) Capitaux mobiliers (épargne) — art. 27 AR")
a_cap = st.checkbox("Le(s) demandeur(s) possède(nt) des capitaux mobiliers")
answers["capital_mobilier_total"] = 0.0
answers["capital_compte_commun"] = False
answers["capital_nb_titulaires"] = 1
answers["capital_conjoint_cotitulaire"] = False
answers["capital_fraction"] = 1.0

if a_cap:
    answers["capital_mobilier_total"] = st.number_input("Montant total capitaux (€)", min_value=0.0, value=0.0, step=100.0)
    compte_commun = st.checkbox("Compte commun ?", value=False)
    answers["capital_compte_commun"] = compte_commun
    if compte_commun:
        answers["capital_nb_titulaires"] = st.number_input("Nombre de titulaires", min_value=1, value=2, step=1)
        if answers["categorie"] == "fam_charge":
            answers["capital_conjoint_cotitulaire"] = st.checkbox("Le conjoint/partenaire est co-titulaire ?", value=True)
    else:
        answers["capital_fraction"] = st.number_input("Part du ménage demandeur (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

# ---------------- Biens immobiliers ----------------
st.divider()
st.subheader("3) Biens immobiliers (RC non indexé) — annuel")
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

# ---------------- Cession ----------------
st.divider()
st.subheader("4) Cession de biens (art. 28 à 32 AR)")
cessions = []
a_ces = st.checkbox("Le ménage demandeur a cédé des biens (10 dernières années)")
answers["cessions"] = []
answers["cession_cas_particulier_37200"] = False
answers["cession_dettes_deductibles"] = 0.0
answers["cession_abatt_cat"] = "cat1"
answers["cession_abatt_mois"] = 0

if a_ces:
    answers["cession_cas_particulier_37200"] = st.checkbox("Cas particulier: tranche immunisée 37.200€", value=False)

    dettes_ok = st.checkbox("Déduire des dettes personnelles (conditions remplies) ?", value=False)
    if dettes_ok:
        answers["cession_dettes_deductibles"] = st.number_input("Dettes déductibles (€)", min_value=0.0, value=0.0, step=100.0)

    answers["cession_abatt_cat"] = st.selectbox("Catégorie d’abattement", ["cat1", "cat2", "cat3"])
    answers["cession_abatt_mois"] = st.number_input("Prorata mois", min_value=0, max_value=12, value=0, step=1)

    nb_c = st.number_input("Nombre de cessions", min_value=0, value=1, step=1)
    for i in range(int(nb_c)):
        st.markdown(f"**Cession {i+1}**")
        val = st.number_input("Valeur vénale (€)", min_value=0.0, value=0.0, step=100.0, key=f"ces_v_{i}")
        usuf = st.checkbox("Usufruit ?", value=False, key=f"ces_u_{i}")
        cessions.append({"valeur_venale": float(val), "usufruit": usuf})
    answers["cessions"] = cessions

# ---------------- Art. 34 cohabitants ----------------
st.divider()
st.subheader("5) Cohabitants admissibles (art. 34) — calcul 'feuilles CPAS'")

st.caption(
    "Pris en compte : partenaire (si NON-demandeur), ascendants/descendants en ligne directe 1er/2e degré "
    "(parent/enfant, grand-parent/petit-enfant). Les autres (frère/sœur, oncle, ami, coloc…) : NON."
)

# Partage UNIQUEMENT si plusieurs enfants/jeunes demandeurs (ex: 2 jeunes)
answers["partage_enfants_jeunes_actif"] = st.checkbox(
    "Partager la part cohabitants entre plusieurs ENFANTS/JEUNES demandeurs (uniquement dans ce cas)",
    value=False
)
answers["nb_enfants_jeunes_demandeurs"] = 1
if answers["partage_enfants_jeunes_actif"]:
    answers["nb_enfants_jeunes_demandeurs"] = st.number_input(
        "Nombre d'enfants/jeunes demandeurs à partager",
        min_value=1, value=2, step=1
    )

cohabitants = []
nb_coh = st.number_input("Nombre de cohabitants à encoder", min_value=0, value=1, step=1)
for i in range(int(nb_coh)):
    st.markdown(f"**Cohabitant {i+1}**")
    c1, c2 = st.columns([2, 1])
    typ = c1.selectbox(
        "Type",
        ["partenaire", "debiteur_direct_1", "debiteur_direct_2", "autre"],
        key=f"coh_type_{i}"
    )
    rev = c2.number_input(
        "Revenus nets annuels (€/an)",
        min_value=0.0, value=0.0, step=100.0,
        key=f"coh_rev_{i}"
    )

    excl = st.checkbox("Ne pas prendre en compte (équité / décision CPAS)", value=False, key=f"coh_excl_{i}")

    # Prestations familiales "en faveur du demandeur" : souvent exonérées → on met par défaut 0 compté
    pf = st.checkbox("PF en faveur du demandeur (perçues par ce cohabitant)", value=False, key=f"coh_pf_{i}")
    pf_a_compter_m = 0.0
    if pf:
        pf_a_compter_m = st.number_input(
            "Montant PF à compter (€/mois) (souvent 0 si exonéré art.22)",
            min_value=0.0, value=0.0, step=10.0,
            key=f"coh_pf_m_{i}"
        )

    cohabitants.append({
        "type": typ,
        "revenu_net_annuel": float(rev),
        "exclure": bool(excl),
        "pf_pour_demandeur": bool(pf),
        "pf_mensuel_a_compter": float(pf_a_compter_m),
    })
answers["cohabitants_art34"] = cohabitants

# ---------------- Avantage en nature ----------------
st.divider()
st.subheader("6) Avantage en nature (art. 33 AR)")
answers["avantage_nature_logement_mensuel"] = st.number_input(
    "Logement payé par un tiers non cohabitant (€/mois) — montant à compter",
    min_value=0.0, value=0.0, step=10.0
)

# ---------------- Calcul ----------------
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
