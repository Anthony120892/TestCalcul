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
    # ⚠️ Attention au nom du fichier : ris_rules.json (1 seul underscore)
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
# CAPITAUX MOBILIERS (art. 27 AR)
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

# ============================================================
# IMMOBILIER (RC non indexé) — multipropriété + indivision + hypo/viager (plafond 50%)
# ============================================================
def immo_monthly_total(biens: list, enfants: int, cfg_immo: dict) -> float:
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
            exo_par_bien = ((exo_bati_total * frac) / nb_bati) if nb_bati > 0 else 0.0
        else:
            exo_par_bien = ((exo_non_bati_total * frac) / nb_non_bati) if nb_non_bati > 0 else 0.0

        base = max(0.0, (rc_part - exo_par_bien) * coeff)

        if b.get("hypotheque", False):
            interets = max(0.0, float(b.get("interets_annuels", 0.0))) * frac
            base -= min(interets, 0.5 * base)

        if b.get("viager", False):
            rente = max(0.0, float(b.get("rente_viagere_annuelle", 0.0))) * frac
            base -= min(rente, 0.5 * base)

        total_annuel += max(0.0, base)

    return total_annuel / 12.0

# ============================================================
# CESSION DE BIENS (art. 28 à 32 AR)
# ============================================================
def cession_biens_monthly(cessions: list,
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

    annuel = 0.0
    if total <= t0_max:
        annuel = 0.0
    else:
        tranche1 = max(0.0, min(total, t1_max) - t1_min)
        annuel += tranche1 * r1
        tranche2 = max(0.0, total - t1_max)
        annuel += tranche2 * r2

    return annuel / 12.0

# ============================================================
# IMMUNISATION SIMPLE (art. 22 §2 AR) — mensuel
# ============================================================
def immunisation_simple_monthly(categorie: str, cfg_immu: dict) -> float:
    return float(cfg_immu.get(categorie, 0.0)) / 12.0

# ============================================================
# COHABITANTS — art.34 “ligne directe” (1er & 2e degré) + partenaire
# + Prestations familiales (forfait 240€ si inconnu, sauf handicap)
# ============================================================
def cohabitants_part_monthly_art34(cohabitants: list, taux_cat1_mensuel: float) -> tuple[float, float, int]:
    """
    Retourne (part_cohabitants_a_compter, prestations_familiales_a_compter, n_pris_en_compte)

    cohabitants: liste de dict:
      {
        "type": "partenaire" | "debiteur_direct_1" | "debiteur_direct_2" | "autre",
        "revenus_annuels": float,
        "exclu_equite": bool,
        "prest_fam_apply": bool,
        "prest_fam_montant": float,   # 0 si inconnu
        "prest_fam_handicap": bool
      }
    """
    admissibles = {"partenaire", "debiteur_direct_1", "debiteur_direct_2"}
    pris = [
        c for c in (cohabitants or [])
        if c.get("type") in admissibles and not bool(c.get("exclu_equite", False))
    ]

    n = len(pris)
    taux_cat1 = max(0.0, float(taux_cat1_mensuel))

    # Revenus cohabitants mensuels
    total_rev_m = sum(max(0.0, float(c.get("revenus_annuels", 0.0))) / 12.0 for c in pris)

    # Part à compter: on garantit 1 taux "catégorie 1 / cohabitant" par majeur pris en compte
    part = max(0.0, total_rev_m - (n * taux_cat1))

    # Prestations familiales (forfait 240€, sauf preuve < 240; suppléments handicap exonérés)
    prest = 0.0
    for c in pris:
        if bool(c.get("prest_fam_apply", False)) and not bool(c.get("prest_fam_handicap", False)):
            m = float(c.get("prest_fam_montant", 0.0))
            prest += m if (m > 0.0 and m < 240.0) else 240.0

    return part, prest, n

# ============================================================
# CALCUL GLOBAL
# ============================================================
def compute_all(answers: dict, engine: dict) -> dict:
    cfg = engine["config"]
    cat = answers.get("categorie", "isole")

    taux_ris_m = float(cfg["ris_rates"].get(cat, 0.0))

    # 1) Revenus demandeur (ANNUEL -> mensuel)
    demandeur_annuel = max(0.0, float(answers.get("demandeur_revenus_annuels", 0.0)))
    revenus_demandeur_m = demandeur_annuel / 12.0

    # 2) Capitaux mobiliers
    cap = capital_mobilier_monthly(
        total_capital=answers.get("capital_mobilier_total", 0.0),
        compte_commun=answers.get("capital_compte_commun", False),
        nb_titulaires=answers.get("capital_nb_titulaires", 1),
        categorie=cat,
        conjoint_compte_commun=answers.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=answers.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    )

    # 3) Immobilier
    immo = immo_monthly_total(
        biens=answers.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )

    # 4) Cession
    cession = cession_biens_monthly(
        cessions=answers.get("cessions", []),
        cas_particulier_tranche_37200=answers.get("cession_cas_particulier_37200", False),
        dettes_deductibles=answers.get("cession_dettes_deductibles", 0.0),
        abatt_cat=answers.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=answers.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    )

    # 5) Cohabitants art.34 (ligne directe 1er/2e degré + partenaire)
    cohab_part_m = 0.0
    prest_fam_m = 0.0
    n_coh = 0

    if cat in ("cohab", "fam_charge"):
        taux_cat1_m = float(cfg["ris_rates"].get("cohab", 0.0))
        cohab_part_m, prest_fam_m, n_coh = cohabitants_part_monthly_art34(
            cohabitants=answers.get("cohabitants_list", []),
            taux_cat1_mensuel=taux_cat1_m
        )

    # 6) Avantage nature
    avantage_nature = max(0.0, float(answers.get("avantage_nature_logement", 0.0)))

    total_avant = revenus_demandeur_m + cap + immo + cession + cohab_part_m + prest_fam_m + avantage_nature

    # Immunisation simple si ressources < taux
    immu_m = 0.0
    if taux_ris_m > 0 and total_avant < taux_ris_m:
        immu_m = immunisation_simple_monthly(cat, cfg["immunisation_simple_annuelle"])

    total_apres = max(0.0, total_avant - immu_m)
    ris = max(0.0, taux_ris_m - total_apres) if taux_ris_m > 0 else 0.0

    return {
        "categorie": cat,
        "enfants_a_charge": int(answers.get("enfants_a_charge", 0)),

        "revenus_demandeur_annuels": demandeur_annuel,
        "revenus_demandeur_mensuels": revenus_demandeur_m,

        "cohabitants_n_pris_en_compte": n_coh,
        "cohabitants_part_a_compter_mensuel": cohab_part_m,
        "prestations_familiales_a_compter_mensuel": prest_fam_m,

        "capitaux_mobiliers_mensuels": cap,
        "immo_mensuels": immo,
        "cession_biens_mensuelle": cession,
        "avantage_nature_logement": avantage_nature,

        "total_ressources_avant_immunisation_simple": total_avant,
        "taux_ris_mensuel": taux_ris_m,
        "immunisation_simple_mensuelle": immu_m,
        "total_ressources_apres_immunisation_simple": total_apres,
        "ris_theorique": ris
    }

# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(page_title="Calcul RIS (prototype)", layout="centered")

engine = load_engine()
cfg = engine["config"]

# ---------------- Logo + titre (header) ----------------
# Mets logo.png dans le même dossier que streamlit_app.py (ou change le chemin)
col1, col2 = st.columns([1, 5])
with col1:
    st.image("logo.png", use_container_width=True)
with col2:
    st.title("Calcul RIS – Prototype (art. 34 – ligne directe 2e degré)")
    st.caption("Annuel demandeur + cohabitants admissibles (partenaire, asc/desc 1er & 2e degré) + prestations familiales + prorata 1er mois")

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
st.subheader("1) Revenus du demandeur (nets) — ANNUELS")
st.caption("Ici : uniquement le demandeur. Les cohabitants se mettent au point 5.")
answers["demandeur_revenus_annuels"] = st.number_input(
    "Total annuel des revenus nets du demandeur (€/an)",
    min_value=0.0, value=0.0, step=100.0
)

# ---------------- Capitaux ----------------
st.divider()
st.subheader("2) Capitaux mobiliers (art. 27 AR)")
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
st.subheader("3) Biens immobiliers (RC non indexé, (RC - exo) × 3)")
biens = []
a_immo = st.checkbox("Le demandeur possède des biens immobiliers")
if a_immo:
    nb_biens = st.number_input("Nombre de biens à encoder", min_value=0, value=1, step=1)
    for i in range(int(nb_biens)):
        st.markdown(f"**Bien {i+1}**")
        habitation_principale = st.checkbox(f"Habitation principale ? (bien {i+1})", value=False, key=f"im_hp_{i}")
        bati = st.checkbox(f"Bien bâti ? (bien {i+1})", value=True, key=f"im_bati_{i}")
        rc_non_indexe = st.number_input(f"RC global NON indexé annuel (bien {i+1})", min_value=0.0, value=0.0, step=50.0, key=f"im_rc_{i}")
        fraction = st.number_input(f"Fraction de droits (0–1) (bien {i+1})", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"im_frac_{i}")

        hypotheque = False
        interets_annuels = 0.0
        viager = False
        rente = 0.0

        if not habitation_principale:
            hypotheque = st.checkbox(f"Hypothèque ? (bien {i+1})", value=False, key=f"im_hyp_{i}")
            if hypotheque:
                interets_annuels = st.number_input(f"Intérêts hypothécaires annuels payés (bien {i+1})", min_value=0.0, value=0.0, step=50.0, key=f"im_int_{i}")

            viager = st.checkbox(f"Acquis en viager ? (bien {i+1})", value=False, key=f"im_vi_{i}")
            if viager:
                rente = st.number_input(f"Rente viagère annuelle payée (bien {i+1})", min_value=0.0, value=0.0, step=50.0, key=f"im_rente_{i}")

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
        cessions.append({"valeur_venale": float(val), "usufruit": usuf})
    answers["cessions"] = cessions

# ---------------- Cohabitants admissibles (art.34) ----------------
st.divider()
st.subheader("5) Cohabitants admissibles (art. 34) — partenaire + ligne directe 1er/2e degré")
answers["cohabitants_list"] = []

if answers["categorie"] in ("cohab", "fam_charge"):
    st.caption("⚠️ Sont pris en compte: partenaire de vie, ascendants/descendants en ligne directe (parent/enfant, grand-parent/petit-enfant).")
    st.caption("Les 'autres' (frère/soeur, oncle, ami, coloc…) ne sont PAS pris en compte.")

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
        pf_apply = st.checkbox(f"Oui, il/elle perçoit des prestations familiales pour le demandeur", value=False, key=f"coh_pf_{i}")

        pf_montant = 0.0
        pf_handicap = False
        if pf_apply:
            pf_montant = st.number_input(
                f"Montant mensuel prouvé (si < 240€) — sinon laisse à 0 pour forfait 240€",
                min_value=0.0, value=0.0, step=10.0, key=f"coh_pf_m_{i}"
            )
            pf_handicap = st.checkbox(
                f"Supplément(s) lié(s) au handicap (exonéré) ?", value=False, key=f"coh_pf_h_{i}"
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
st.subheader("6) Avantage en nature (art. 33 AR)")
answers["avantage_nature_logement"] = st.number_input(
    "Logement payé par un tiers non cohabitant (€/mois) – montant à compter",
    min_value=0.0, value=0.0, step=10.0
)

# ---------------- Calcul ----------------
st.divider()
if st.button("Calculer le RIS"):
    res = compute_all(answers, engine)
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

    st.write("### Détail (mensuel)")
    res["date_demande"] = str(demande_date)
    res["jours_dans_mois"] = dim
    res["jours_restants_inclus"] = jours_restants
    res["prorata_premier_mois"] = prorata
    res["ris_premier_mois_prorata"] = ris_m1
    res["ris_mois_suivants"] = ris_m
    st.json(res)
