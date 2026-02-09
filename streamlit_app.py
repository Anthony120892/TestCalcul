import json
import os
import calendar
from datetime import date
import streamlit as st

# ============================================================
# CONFIG PAR DÉFAUT — fusionnable avec ris_rules.json
# ============================================================
DEFAULT_ENGINE = {
    "version": "1.0",
    "config": {
        # Taux RIS (mensuel) — à encoder dans la sidebar
        "ris_rates": {"cohab": 0.0, "isole": 0.0, "fam_charge": 0.0},

        # Immunisation simple (art. 22 §2 AR) — annuel
        "immunisation_simple_annuelle": {"cohab": 155.0, "isole": 250.0, "fam_charge": 310.0},

        # Capitaux mobiliers (art. 27 AR) — seuils annuels
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

        # Cession de biens (art. 28 à 32 AR)
        "cession": {
            "tranche_immunisee": 37200.0,
            "usufruit_ratio": 0.40,
            "abattements_annuels": {"cat1": 1250.0, "cat2": 2000.0, "cat3": 2500.0}
        }
    }
}

# ============================================================
# UTILITAIRES
# ============================================================
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def deep_merge(base: dict, override: dict) -> dict:
    """Fusion récursive: override écrase base, sans perdre les clés manquantes."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def normalize_engine(raw: dict) -> dict:
    """
    Rend ton ris_rules.json minimal compatible avec le moteur.
    - gère les anciens noms (exonerations -> immo)
    - complète les sections manquantes
    """
    raw = raw or {}
    engine = deep_merge(DEFAULT_ENGINE, raw)
    cfg = engine["config"]

    # Compat: si l’utilisateur n’a que "exonerations", on mappe vers "immo"
    if "exonerations" in cfg and "immo" in cfg:
        exo = cfg["exonerations"]
        cfg["immo"]["bati_base"] = float(exo.get("bati_base", cfg["immo"]["bati_base"]))
        cfg["immo"]["bati_par_enfant"] = float(exo.get("bati_par_enfant", cfg["immo"]["bati_par_enfant"]))
        cfg["immo"]["non_bati_base"] = float(exo.get("non_bati_base", cfg["immo"]["non_bati_base"]))

    # Sécurité: forcer types numériques
    for k in ("cohab", "isole", "fam_charge"):
        cfg["ris_rates"][k] = float(cfg["ris_rates"].get(k, 0.0))
        cfg["immunisation_simple_annuelle"][k] = float(cfg["immunisation_simple_annuelle"].get(k, 0.0))

    return engine

def load_engine() -> dict:
    # ⚠️ Nom attendu: ris_rules.json (un seul underscore)
    fname = "ris_rules.json"
    if os.path.exists(fname):
        with open(fname, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return normalize_engine(raw)
    return normalize_engine(DEFAULT_ENGINE)

# ---- Prorata 1er mois ----
def days_in_month(d: date) -> int:
    return calendar.monthrange(d.year, d.month)[1]

def prorata_remaining_from_date(d: date) -> float:
    """
    Prorata basé sur les jours RESTANTS dans le mois, jour de la demande inclus.
    ex: demande le 10 -> paiement du 10 au dernier jour inclus.
    """
    dim = days_in_month(d)
    remaining = dim - d.day + 1
    return remaining / dim

def ris_first_month_amount(ris_mensuel: float, demande_date: date) -> float:
    return max(0.0, float(ris_mensuel)) * prorata_remaining_from_date(demande_date)

# ============================================================
# CAPITAUX MOBILIERS (art. 27 AR) — calc annuel via seuils annuels
# ============================================================
def capital_mobilier_monthly(total_capital: float,
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
        annuel = 0.0
    else:
        tranche1 = max(0.0, min(adj_total, t1_max) - t1_min)
        annuel += tranche1 * r1
        tranche2 = max(0.0, adj_total - t1_max)
        annuel += tranche2 * r2

    return annuel / 12.0

def capital_mobilier_annual(**kwargs) -> float:
    return float(capital_mobilier_monthly(**kwargs)) * 12.0

# ============================================================
# IMMOBILIER — retourne MENSUEL -> on annualise pour le mode CPAS
# - RC non indexé: (RC - exo) * 3
# - Si revenus locatifs > RC-calculé, on prend les loyers
# - Habitation principale exclue
# - Nue-propriété: non comptée (on ignore le bien)
# - Indivision (fraction) prise en compte
# - Hypothèque/viager: déduction plafonnée à 50% du RC-calculé
# ============================================================
def immo_monthly_total(biens: list, enfants: int, cfg_immo: dict) -> float:
    biens_countes = [
        b for b in (biens or [])
        if (not b.get("habitation_principale", False)) and (not b.get("nue_propriete", False))
    ]

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
            exo_par_bien = ((exo_bati_total * frac) / nb_bati) if nb_bati > 0 else 0.0
        else:
            exo_par_bien = ((exo_non_bati_total * frac) / nb_non_bati) if nb_non_bati > 0 else 0.0

        base_rc = max(0.0, (rc_part - exo_par_bien) * coeff)

        if b.get("hypotheque", False):
            interets = max(0.0, float(b.get("interets_annuels", 0.0))) * frac
            base_rc -= min(interets, 0.5 * base_rc)

        if b.get("viager", False):
            rente = max(0.0, float(b.get("rente_viagere_annuelle", 0.0))) * frac
            base_rc -= min(rente, 0.5 * base_rc)

        base_rc = max(0.0, base_rc)

        bien_loue = bool(b.get("bien_loue", False))
        loyers_annuels = max(0.0, float(b.get("loyers_annuels", 0.0))) * frac
        base_finale = max(base_rc, loyers_annuels) if bien_loue else base_rc

        total_annuel += base_finale

    return total_annuel / 12.0

def immo_annual_total(biens: list, enfants: int, cfg_immo: dict) -> float:
    return float(immo_monthly_total(biens=biens, enfants=enfants, cfg_immo=cfg_immo)) * 12.0

# ============================================================
# CESSION DE BIENS (art. 28 à 32 AR) — retourne mensuel -> annualisé
# ============================================================
def cession_biens_monthly(cessions: list,
                          cas_particulier_tranche_37200: bool,
                          dettes_deductibles: float,
                          abatt_cat: str,
                          abatt_mois_prorata: int,
                          cfg_cession: dict,
                          cfg_cap: dict) -> float:
    total = 0.0
    for c in (cessions or []):
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

    annuel = 0.0
    if total <= t0_max:
        annuel = 0.0
    else:
        tranche1 = max(0.0, min(total, t1_max) - t1_min)
        annuel += tranche1 * r1
        tranche2 = max(0.0, total - t1_max)
        annuel += tranche2 * r2

    return annuel / 12.0

def cession_biens_annual(**kwargs) -> float:
    return float(cession_biens_monthly(**kwargs)) * 12.0

# ============================================================
# COHABITANTS — MODE OFFICIEL CPAS (ANNUEL)
# art.34 (ligne directe 1er & 2e degré) + partenaire
# - part_total_annuel = max(0, somme_revenus_annuels - N*taux_cat1_annuel)
# - partage entre nb_demandeurs_menage (ex: 2 demandeurs)
# + Prestations familiales: forfait 240€/mois si inconnu, sauf handicap
#   -> ici, ON COMPTE en ANNUEL (mensuel * 12)
# ============================================================
def cohabitants_part_annual_art34(cohabitants: list,
                                 taux_cat1_annuel: float,
                                 nb_demandeurs_menage: int) -> tuple[float, float, int, float]:
    """
    Retourne:
      (part_par_demandeur_annuel, prestations_familiales_annuel, n_pris_en_compte, part_totale_avant_partage_annuel)
    """
    admissibles = {"partenaire", "debiteur_direct_1", "debiteur_direct_2"}
    pris = [
        c for c in (cohabitants or [])
        if c.get("type") in admissibles and not bool(c.get("exclu_equite", False))
    ]

    n = len(pris)
    taux_cat1_a = max(0.0, float(taux_cat1_annuel))

    total_rev_a = sum(max(0.0, float(c.get("revenus_annuels", 0.0))) for c in pris)
    part_total_a = max(0.0, total_rev_a - (n * taux_cat1_a))

    nb = max(1, int(nb_demandeurs_menage))
    part_par_dem_a = part_total_a / nb

    pf_annuel = 0.0
    for c in pris:
        # Par défaut, on NE COMPTE PAS. Si tu coches "prestations familiales à compter", on annualise.
        if bool(c.get("prest_fam_apply", False)) and not bool(c.get("prest_fam_handicap", False)):
            m = float(c.get("prest_fam_montant", 0.0))
            pf_m = m if (m > 0.0 and m < 240.0) else 240.0
            pf_annuel += pf_m * 12.0

    return part_par_dem_a, pf_annuel, n, part_total_a

# ============================================================
# CALCUL GLOBAL — MODE OFFICIEL CPAS (ANNUEL)
# Tout est calculé en ANNUEL, et on divise par 12 à la fin.
# ============================================================
def compute_all_officiel_cpas_annuel(answers: dict, engine: dict) -> dict:
    cfg = engine["config"]
    cat = answers.get("categorie", "isole")

    # Taux RIS annuel (dérivé du mensuel)
    taux_ris_m = float(cfg["ris_rates"].get(cat, 0.0))
    taux_ris_a = taux_ris_m * 12.0

    # 1) Revenus ménage demandeur (annuel) — (personne seule ou couple demandeur) => tu mets le total ici
    revenus_demandeur_a = max(0.0, float(answers.get("demandeur_revenus_annuels", 0.0)))

    # 2) Capitaux mobiliers (annuel)
    cap_a = capital_mobilier_annual(
        total_capital=answers.get("capital_mobilier_total", 0.0),
        compte_commun=answers.get("capital_compte_commun", False),
        nb_titulaires=answers.get("capital_nb_titulaires", 1),
        categorie=cat,
        conjoint_compte_commun=answers.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=answers.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    )

    # 3) Immobilier (annuel)
    immo_a = immo_annual_total(
        biens=answers.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )

    # 4) Cession (annuel)
    cession_a = cession_biens_annual(
        cessions=answers.get("cessions", []),
        cas_particulier_tranche_37200=answers.get("cession_cas_particulier_37200", False),
        dettes_deductibles=answers.get("cession_dettes_deductibles", 0.0),
        abatt_cat=answers.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=answers.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    )

    # 5) Cohabitants art.34 (annuel)
    cohab_part_a = 0.0
    prest_fam_a = 0.0
    n_coh = 0
    part_total_coh_a = 0.0

    if cat in ("cohab", "fam_charge"):
        # "taux catégorie 1" = taux cohab (mensuel) -> annualisé
        taux_cat1_a = float(cfg["ris_rates"].get("cohab", 0.0)) * 12.0
        nb_dem = int(answers.get("nb_demandeurs_menage", 1))

        cohab_part_a, prest_fam_a, n_coh, part_total_coh_a = cohabitants_part_annual_art34(
            cohabitants=answers.get("cohabitants_list", []),
            taux_cat1_annuel=taux_cat1_a,
            nb_demandeurs_menage=nb_dem
        )

    # 6) Avantage en nature (annuel)
    avantage_nature_m = max(0.0, float(answers.get("avantage_nature_logement", 0.0)))
    avantage_nature_a = avantage_nature_m * 12.0

    total_avant_a = revenus_demandeur_a + cap_a + immo_a + cession_a + cohab_part_a + prest_fam_a + avantage_nature_a

    # Immunisation simple (annuelle) si ressources < taux
    immu_a = 0.0
    if taux_ris_a > 0 and total_avant_a < taux_ris_a:
        immu_a = float(cfg["immunisation_simple_annuelle"].get(cat, 0.0))

    total_apres_a = max(0.0, total_avant_a - immu_a)
    ris_a = max(0.0, taux_ris_a - total_apres_a) if taux_ris_a > 0 else 0.0

    # Conversion mensuelle uniquement à la fin
    revenus_demandeur_m = revenus_demandeur_a / 12.0
    cap_m = cap_a / 12.0
    immo_m = immo_a / 12.0
    cession_m = cession_a / 12.0
    cohab_part_m = cohab_part_a / 12.0
    prest_fam_m = prest_fam_a / 12.0
    avantage_nature_m = avantage_nature_a / 12.0
    total_avant_m = total_avant_a / 12.0
    immu_m = immu_a / 12.0
    total_apres_m = total_apres_a / 12.0
    ris_m = ris_a / 12.0

    return {
        # Meta
        "mode_calcul": "OFFICIEL_CPAS_ANNUEL",
        "categorie": cat,
        "enfants_a_charge": int(answers.get("enfants_a_charge", 0)),
        "nb_demandeurs_menage_pour_partage": int(answers.get("nb_demandeurs_menage", 1)),

        # Détails ANNUELS (comme tes feuilles)
        "revenus_demandeur_annuels": revenus_demandeur_a,
        "capitaux_mobiliers_annuels": cap_a,
        "immo_annuels": immo_a,
        "cession_biens_annuelle": cession_a,
        "cohabitants_n_pris_en_compte": n_coh,
        "cohabitants_part_totale_avant_partage_annuel": part_total_coh_a,
        "cohabitants_part_a_compter_par_demandeur_annuel": cohab_part_a,
        "prestations_familiales_a_compter_annuel": prest_fam_a,
        "avantage_nature_logement_annuel": avantage_nature_a,

        "total_ressources_avant_immunisation_simple_annuel": total_avant_a,
        "taux_ris_annuel": taux_ris_a,
        "immunisation_simple_annuelle": immu_a,
        "total_ressources_apres_immunisation_simple_annuel": total_apres_a,
        "ris_theorique_annuel": ris_a,

        # Détails MENSUELS (pour affichage)
        "revenus_demandeur_mensuels": revenus_demandeur_m,
        "capitaux_mobiliers_mensuels": cap_m,
        "immo_mensuels": immo_m,
        "cession_biens_mensuelle": cession_m,
        "cohabitants_part_a_compter_par_demandeur_mensuel": cohab_part_m,
        "prestations_familiales_a_compter_mensuel": prest_fam_m,
        "avantage_nature_logement_mensuel": avantage_nature_m,

        "total_ressources_avant_immunisation_simple": total_avant_m,
        "taux_ris_mensuel": taux_ris_m,
        "immunisation_simple_mensuelle": immu_m,
        "total_ressources_apres_immunisation_simple": total_apres_m,
        "ris_theorique": ris_m
    }

# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(page_title="Calcul RIS (CPAS officiel annuel)", layout="centered")

engine = load_engine()
cfg = engine["config"]

# ---------------- Logo + titre ----------------
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", use_container_width=True)
with col2:
    st.title("Calcul RIS – Prototype (CPAS)")
    st.caption("MODE OFFICIEL CPAS: calcul 100% ANNUEL puis /12 à la fin (comme tes feuilles) + prorata 1er mois")

# ---------------- Sidebar paramètres ----------------
with st.sidebar:
    st.image("logo.png", use_container_width=True)

    st.subheader("Paramètres")
    st.write("**Taux RIS mensuels officiels** :")
    cfg["ris_rates"]["cohab"] = st.number_input("Taux RIS cohabitant (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["cohab"]))
    cfg["ris_rates"]["isole"] = st.number_input("Taux RIS isolé (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["isole"]))
    cfg["ris_rates"]["fam_charge"] = st.number_input("Taux RIS famille à charge (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["fam_charge"]))

    st.divider()
    st.write("**Immunisation simple (€/an)**")
    cfg["immunisation_simple_annuelle"]["cohab"] = st.number_input("Immunisation simple cohab (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["cohab"]))
    cfg["immunisation_simple_annuelle"]["isole"] = st.number_input("Immunisation simple isolé (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["isole"]))
    cfg["immunisation_simple_annuelle"]["fam_charge"] = st.number_input("Immunisation simple fam. charge (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["fam_charge"]))

# ---------------- Answers ----------------
answers = {}
answers["categorie"] = st.selectbox("Catégorie", ["cohab", "isole", "fam_charge"])
answers["enfants_a_charge"] = st.number_input("Enfants à charge", min_value=0, value=0, step=1)

# ---------------- Date demande (pour prorata) ----------------
st.divider()
st.subheader("Date de la demande (pour calcul du 1er mois)")
demande_date = st.date_input("Date de la demande", value=date.today())

# ---------------- Revenus demandeur (ANNUEL) ----------------
st.divider()
st.subheader("1) Revenus du ménage demandeur (nets) — ANNUELS")
st.caption("Si un couple fait la demande: additionne les revenus des 2 demandeurs ici (total annuel).")
answers["demandeur_revenus_annuels"] = st.number_input(
    "Total annuel des revenus nets du ménage demandeur (€/an)",
    min_value=0.0, value=0.0, step=100.0
)

# ---------------- Capitaux ----------------
st.divider()
st.subheader("2) Capitaux mobiliers (épargne) (art. 27 AR)")
a_cap = st.checkbox("Le demandeur possède des capitaux mobiliers")
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
        answers["capital_nb_titulaires"] = st.number_input("Nombre de titulaires du compte", min_value=1, value=2, step=1)
        if answers["categorie"] == "fam_charge":
            answers["capital_conjoint_cotitulaire"] = st.checkbox("Le conjoint/partenaire est aussi co-titulaire ?", value=True)
    else:
        answers["capital_fraction"] = st.number_input("Part du demandeur (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

# ---------------- Biens immobiliers ----------------
st.divider()
st.subheader("3) Biens immobiliers")
st.caption("Calcul: (RC non indexé – exonération) × 3. Si bien loué et loyers > RC×3, on prend les loyers.")
st.caption("Nue-propriété: non comptée. Habitation principale: non comptée.")
biens = []
a_immo = st.checkbox("Le demandeur possède des biens immobiliers")
if a_immo:
    nb_biens = st.number_input("Nombre de biens à encoder", min_value=0, value=1, step=1)
    for i in range(int(nb_biens)):
        st.markdown(f"**Bien {i+1}**")

        c1, c2, c3 = st.columns([1, 1, 1])
        habitation_principale = c1.checkbox("Habitation principale ?", value=False, key=f"im_hp_{i}")
        nue_propriete = c2.checkbox("Nue-propriété ?", value=False, key=f"im_np_{i}")
        bati = c3.checkbox("Bien bâti ?", value=True, key=f"im_bati_{i}")

        rc_non_indexe = st.number_input(
            f"RC global NON indexé annuel (bien {i+1})",
            min_value=0.0, value=0.0, step=50.0, key=f"im_rc_{i}"
        )
        fraction = st.number_input(
            f"Fraction de droits (0–1) (bien {i+1})",
            min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"im_frac_{i}"
        )

        bien_loue = False
        loyers_annuels = 0.0
        hypotheque = False
        interets_annuels = 0.0
        viager = False
        rente = 0.0

        if (not habitation_principale) and (not nue_propriete):
            bien_loue = st.checkbox(f"Bien loué ? (bien {i+1})", value=False, key=f"im_loue_{i}")
            if bien_loue:
                loyers_annuels = st.number_input(
                    f"Loyers perçus annuels (bruts) (bien {i+1})",
                    min_value=0.0, value=0.0, step=100.0, key=f"im_loyers_{i}"
                )

            hypotheque = st.checkbox(f"Hypothèque ? (bien {i+1})", value=False, key=f"im_hyp_{i}")
            if hypotheque:
                interets_annuels = st.number_input(
                    f"Intérêts hypothécaires annuels payés (bien {i+1})",
                    min_value=0.0, value=0.0, step=50.0, key=f"im_int_{i}"
                )

            viager = st.checkbox(f"Acquis en viager ? (bien {i+1})", value=False, key=f"im_vi_{i}")
            if viager:
                rente = st.number_input(
                    f"Rente viagère annuelle payée (bien {i+1})",
                    min_value=0.0, value=0.0, step=50.0, key=f"im_rente_{i}"
                )

        biens.append({
            "habitation_principale": bool(habitation_principale),
            "nue_propriete": bool(nue_propriete),
            "bati": bool(bati),
            "rc_non_indexe": float(rc_non_indexe),
            "fraction_droits": float(fraction),

            "bien_loue": bool(bien_loue),
            "loyers_annuels": float(loyers_annuels),

            "hypotheque": bool(hypotheque),
            "interets_annuels": float(interets_annuels),
            "viager": bool(viager),
            "rente_viagere_annuelle": float(rente)
        })
answers["biens_immobiliers"] = biens

# ---------------- Cession de biens ----------------
st.divider()
st.subheader("4) Cession de biens (art. 28 à 32 AR)")
cessions = []
a_ces = st.checkbox("Le demandeur a cédé des biens dans les 10 dernières années")
answers["cessions"] = []
answers["cession_cas_particulier_37200"] = False
answers["cession_dettes_deductibles"] = 0.0
answers["cession_abatt_cat"] = "cat1"
answers["cession_abatt_mois"] = 0

if a_ces:
    answers["cession_cas_particulier_37200"] = st.checkbox("Cas particulier: tranche immunisée 37.200€ applicable", value=False)

    dettes_ok = st.checkbox("Déduire des dettes personnelles (conditions remplies) ?", value=False)
    if dettes_ok:
        answers["cession_dettes_deductibles"] = st.number_input("Montant total des dettes déductibles (€)", min_value=0.0, value=0.0, step=100.0)

    answers["cession_abatt_cat"] = st.selectbox("Catégorie d’abattement", ["cat1", "cat2", "cat3"])
    answers["cession_abatt_mois"] = st.number_input("Nombre de mois (prorata)", min_value=0, max_value=12, value=0, step=1)

    nb_c = st.number_input("Nombre de cessions à encoder", min_value=0, value=1, step=1)
    for i in range(int(nb_c)):
        st.markdown(f"**Cession {i+1}**")
        val = st.number_input(f"Valeur vénale (€) (cession {i+1})", min_value=0.0, value=0.0, step=100.0, key=f"ces_v_{i}")
        usuf = st.checkbox(f"Cession d’usufruit ? (cession {i+1})", value=False, key=f"ces_u_{i}")
        cessions.append({"valeur_venale": float(val), "usufruit": bool(usuf)})
    answers["cessions"] = cessions

# ---------------- Cohabitants admissibles (art.34) + partage ----------------
st.divider()
st.subheader("5) Cohabitants admissibles (art. 34) — + partage entre demandeurs")
answers["cohabitants_list"] = []
answers["nb_demandeurs_menage"] = 1

if answers["categorie"] in ("cohab", "fam_charge"):
    st.caption("Pris en compte: partenaire, ascendants/descendants en ligne directe (parent/enfant, grand-parent/petit-enfant).")
    st.caption("Les 'autres' (frère/soeur, oncle, ami, coloc…) ne sont PAS pris en compte.")
    st.caption("⚠️ Si plusieurs demandeurs DIS dans le ménage, on partage la part calculée (comme tes feuilles).")

    answers["nb_demandeurs_menage"] = st.number_input(
        "Nombre de demandeurs/bénéficiaires DIS à partager dans le ménage",
        min_value=1, value=1, step=1
    )

    nb_coh = st.number_input("Nombre de cohabitants à encoder", min_value=0, value=1, step=1)
    cohs = []

    for i in range(int(nb_coh)):
        st.markdown(f"**Cohabitant {i+1}**")
        c1, c2 = st.columns([2, 1])

        typ = c1.selectbox(
            f"Type (cohabitant {i+1})",
            ["partenaire", "debiteur_direct_1", "debiteur_direct_2", "autre"],
            key=f"coh_typ_{i}"
        )

        revenus_ann = c2.number_input(
            f"Revenus nets annuels (€/an) (cohabitant {i+1})",
            min_value=0.0, value=0.0, step=100.0, key=f"coh_rev_{i}"
        )

        exclu = st.checkbox(
            f"Ne pas prendre en compte (équité / décision CPAS) (cohabitant {i+1})",
            value=False, key=f"coh_eq_{i}"
        )

        st.markdown("**Prestations familiales en faveur du demandeur (perçues par ce cohabitant)**")
        st.caption("⚠️ Ne coche que si ces prestations doivent être COMPTÉES (sinon laisse décoché).")
        pf_apply = st.checkbox(
            "Oui, prestations familiales À COMPTER pour le demandeur",
            value=False, key=f"coh_pf_{i}"
        )

        pf_montant = 0.0
        pf_handicap = False
        if pf_apply:
            pf_montant = st.number_input(
                "Montant mensuel prouvé (si < 240€) — sinon laisse à 0 pour forfait 240€",
                min_value=0.0, value=0.0, step=10.0, key=f"coh_pf_m_{i}"
            )
            pf_handicap = st.checkbox(
                "Supplément(s) lié(s) au handicap (exonéré) ?",
                value=False, key=f"coh_pf_h_{i}"
            )

        cohs.append({
            "type": typ,
            "revenus_annuels": float(revenus_ann),
            "exclu_equite": bool(exclu),
            "prest_fam_apply": bool(pf_apply),
            "prest_fam_montant": float(pf_montant),
            "prest_fam_handicap": bool(pf_handicap)
        })

    answers["cohabitants_list"] = cohs
else:
    st.info("Catégorie 'isolé' : pas de cohabitants à encoder ✅")

# ---------------- Avantage en nature ----------------
st.divider()
st.subheader("6) Avantage en nature")
answers["avantage_nature_logement"] = st.number_input(
    "Logement payé par un tiers non cohabitant (€/mois) – montant à compter",
    min_value=0.0, value=0.0, step=10.0
)

# ---------------- Calcul ----------------
st.divider()
if st.button("Calculer le RIS"):
    res = compute_all_officiel_cpas_annuel(answers, engine)
    ris_m = float(res["ris_theorique"])

    # Prorata 1er mois
    prorata = prorata_remaining_from_date(demande_date)
    ris_m1 = ris_first_month_amount(ris_m, demande_date)
    dim = days_in_month(demande_date)
    jours_restants = dim - demande_date.day + 1

    st.success("Calcul terminé ✅")

    st.metric("RIS mensuel normal (€/mois)", f"{ris_m:.2f}")

    st.write("### Versement")
    st.metric("RIS du 1er mois (prorata) (mois en cours)", f"{ris_m1:.2f}")
    st.caption(f"Prorata = {jours_restants}/{dim} = {prorata:.6f} (jour de demande inclus)")
    st.write(f"**Mois suivants :** {ris_m:.2f} €/mois (RIS complet)")

    st.write("### Détail (CPAS officiel — ANNUEL puis mensuel)")
    res["date_demande"] = str(demande_date)
    res["jours_dans_mois"] = dim
    res["jours_restants_inclus"] = jours_restants
    res["prorata_premier_mois"] = prorata
    res["ris_premier_mois_prorata"] = ris_m1
    res["ris_mois_suivants"] = ris_m
    st.json(res)
