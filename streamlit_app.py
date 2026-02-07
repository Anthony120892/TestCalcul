import json
import os
import streamlit as st

# ============================================================
# CONFIG PAR DÉFAUT (si ris_rules.json absent)
# ============================================================
DEFAULT_ENGINE = {
    "config": {
        "ris_rates": {  # à compléter dans la sidebar
            "cohab": 0.0,
            "isole": 0.0,
            "fam_charge": 0.0
        },
        "immunisation_simple_annuelle": {  # art. 22 §2 AR
            "cohab": 155.0,
            "isole": 250.0,
            "fam_charge": 310.0
        },
        "capital_mobilier": {  # art. 27 AR
            "t0": 6200.0,
            "t1": 12500.0,
            "rate_1": 0.06,
            "rate_2": 0.10
        },
        "immo": {  # règles RC (non indexé)
            "bati_base": 750.0,
            "bati_par_enfant": 125.0,
            "non_bati_base": 30.0,
            "coeff_rc": 3.0
        },
        "socio_prof": {  # immunisation double / exo socio-pro
            "max_mensuel": 274.82,          # à adapter si indexé
            "artistique_annuel": 3297.80,   # 274,82*12 (à adapter si indexé)
        },
        "cession": {  # art. 28-32 AR
            "tranche_immunisee": 37200.0,
            "usufruit_ratio": 0.40,
            "abattements_annuels": {  # cat. 1/2/3
                "cat1": 1250.0,
                "cat2": 2000.0,
                "cat3": 2500.0
            }
        },
        "ale": {  # mention des exclusions ALE (simplifié)
            "exon_mensuelle": 4.10
        }
    }
}


# ============================================================
# UTILITAIRES
# ============================================================
def load_engine():
    if os.path.exists("ris_rules.json"):
        with open("ris_rules.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_ENGINE


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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
    """
    Règle : 0% jusque 6.200€, 6% entre 6.200-12.500, 10% au-delà.
    Compte commun:
      - fraction = 1/nb_titulaires
      - si categorie = fam_charge ET conjoint_compte_commun = True : fraction = 2/nb_titulaires
    Sinon: fraction = part_fraction_custom (0-1)
    On applique la fraction au capital ET aux seuils, ce qui revient à calculer sur total * fraction.
    """
    total_capital = max(0.0, float(total_capital))

    if compte_commun:
        nb = max(1, int(nb_titulaires))
        numerator = 2 if (categorie == "fam_charge" and conjoint_compte_commun) else 1
        fraction = numerator / nb
    else:
        fraction = clamp01(part_fraction_custom)

    adj_total = total_capital * fraction

    t0 = cfg_cap["t0"] * fraction
    t1 = cfg_cap["t1"] * fraction
    r1 = cfg_cap["rate_1"]
    r2 = cfg_cap["rate_2"]

    annuel = 0.0
    if adj_total <= t0:
        annuel = 0.0
    else:
        tranche1 = max(0.0, min(adj_total, t1) - t0)
        annuel += tranche1 * r1
        tranche2 = max(0.0, adj_total - t1)
        annuel += tranche2 * r2

    return annuel / 12.0


# ============================================================
# BIENS IMMOBILIERS (RC non indexé, (RC - exo) * 3)
# + multipropriété (exo divisé par nb de biens)
# + indivision (RC et exo * fraction)
# + hypothèque et viager (déduction plafonnée à 50%)
# ============================================================
def immo_monthly_total(biens: list, enfants: int, cfg_immo: dict) -> float:
    """
    biens: liste de dict
      - habitation_principale: bool
      - bati: bool
      - rc_non_indexe: float (annuel)
      - fraction_droits: float (0-1)  # indivision/pleine prop/usufruit
      - hypotheque: bool
      - interets_annuels: float
      - viager: bool
      - rente_viagere_annuelle: float
    """

    # On ne compte que les biens non "habitation principale"
    biens_countes = [b for b in biens if not b.get("habitation_principale", False)]

    # Pour multipropriété : exonération divisée par le nombre de biens (par catégorie bâti / non bâti)
    nb_bati = sum(1 for b in biens_countes if b.get("bati", True))
    nb_non_bati = sum(1 for b in biens_countes if not b.get("bati", True))
    nb_bati = max(1, nb_bati) if any(b.get("bati", True) for b in biens_countes) else 0
    nb_non_bati = max(1, nb_non_bati) if any((not b.get("bati", True)) for b in biens_countes) else 0

    base_exo_bati_total = cfg_immo["bati_base"] + cfg_immo["bati_par_enfant"] * max(0, int(enfants))
    base_exo_non_bati_total = cfg_immo["non_bati_base"]

    total_annuel = 0.0

    for b in biens_countes:
        bati = bool(b.get("bati", True))
        rc = max(0.0, float(b.get("rc_non_indexe", 0.0)))
        frac = clamp01(b.get("fraction_droits", 1.0))

        # Indivision: RC * fraction avant calcul
        rc_part = rc * frac

        # Exo (par bien) : exo_total * fraction, puis / nb biens de ce type
        if bati:
            exo_total_part = base_exo_bati_total * frac
            exo_par_bien = exo_total_part / (nb_bati if nb_bati > 0 else 1)
        else:
            exo_total_part = base_exo_non_bati_total * frac
            exo_par_bien = exo_total_part / (nb_non_bati if nb_non_bati > 0 else 1)

        base = max(0.0, (rc_part - exo_par_bien) * cfg_immo["coeff_rc"])  # annuel à prendre en compte (avant réductions)

        # Hypothèque : réduction = intérêts * fraction, plafonnée à 50% du montant à prendre en considération
        if b.get("hypotheque", False):
            interets = max(0.0, float(b.get("interets_annuels", 0.0))) * frac
            base -= min(interets, 0.5 * base)

        # Viager : réduction = rente * fraction, plafonnée à 50% du montant à prendre en considération
        if b.get("viager", False):
            rente = max(0.0, float(b.get("rente_viagere_annuelle", 0.0))) * frac
            base -= min(rente, 0.5 * base)

        total_annuel += max(0.0, base)

    return total_annuel / 12.0


# ============================================================
# CESSION DE BIENS (art. 28 à 32 AR)
# - Calcul comme capitaux mobiliers sur valeur vénale
# - Indivision : valeur * fraction (ou fraction demandeur+conjoint si fam_charge)
# - Usufruit : 40% de la pleine propriété
# - Tranche immunisée 37.200 (cas particulier) pondérée par fraction
# - Déduction dettes (option) + abattement annuel proratisé
# ============================================================
def cession_biens_monthly(cessions: list,
                          categorie: str,
                          conjoint_fraction_in_division: bool,
                          fraction_demandeur: float,
                          fraction_demandeur_et_conjoint: float,
                          cas_particulier_tranche_37200: bool,
                          dettes_deductibles: float,
                          abatt_cat: str,
                          abatt_mois_prorata: int,
                          cfg_cession: dict,
                          cfg_cap: dict) -> float:
    """
    cessions: liste dict
      - valeur_venale: float
      - usufruit: bool
      - en_indivision: bool
      - fraction_droits: float (0-1) si indivision (sinon 1)
    Règle: si plusieurs cessions -> tranches une fois sur total des cessions
    (et séparé des capitaux mobiliers)
    """

    total = 0.0
    for c in cessions:
        v = max(0.0, float(c.get("valeur_venale", 0.0)))
        if c.get("usufruit", False):
            v *= float(cfg_cession["usufruit_ratio"])

        if c.get("en_indivision", False):
            frac = clamp01(c.get("fraction_droits", 1.0))
            v *= frac

        # Cas "fam_charge + conjoint/partenaire en indivision": on permet d'utiliser une fraction globale (demandeur+conjoint)
        if categorie == "fam_charge" and conjoint_fraction_in_division:
            v *= clamp01(fraction_demandeur_et_conjoint)
        else:
            v *= clamp01(fraction_demandeur)

        total += v

    # Déduction des dettes personnelles liées à la cession (si conditions remplies -> c'est toi qui coches)
    total = max(0.0, total - max(0.0, float(dettes_deductibles)))

    # Tranche immunisée 37.200 (cas particulier) pondérée (déjà pondérée via fractions ci-dessus)
    if cas_particulier_tranche_37200:
        total = max(0.0, total - float(cfg_cession["tranche_immunisee"]))

    # Abattement annuel (cat. 1/2/3) proratisé
    abatt_annuel = float(cfg_cession["abattements_annuels"].get(abatt_cat, 0.0))
    mois = max(0, min(12, int(abatt_mois_prorata)))
    abatt_prorata = abatt_annuel * (mois / 12.0)
    total = max(0.0, total - abatt_prorata)

    # Ensuite, calcul comme capitaux mobiliers (mais sur "valeur vénale")
    # Ici pas de compte commun: on calc directement sur total (seuils non fractionnés car déjà pondérés)
    t0 = cfg_cap["t0"]
    t1 = cfg_cap["t1"]
    r1 = cfg_cap["rate_1"]
    r2 = cfg_cap["rate_2"]

    annuel = 0.0
    if total <= t0:
        annuel = 0.0
    else:
        tranche1 = max(0.0, min(total, t1) - t0)
        annuel += tranche1 * r1
        tranche2 = max(0.0, total - t1)
        annuel += tranche2 * r2

    return annuel / 12.0


# ============================================================
# EXONERATIONS SOCIO-PRO / ARTISTIQUE (version "moteur" simple)
# ============================================================
def revenus_mensuels_total(revenus: list, cfg_soc: dict) -> float:
    """
    revenus: liste dict
      - montant_mensuel: float
      - type: "standard" | "socio_prof" | "etudiant" | "artistique_irregulier" | "ale"
      - eligible: bool (pour socio_prof/etudiant/artistique)
      - ale_part_excedentaire: float (si type=ale) -> part > 4,10 à compter (simplifié)
    Règles:
      - socio_prof / etudiant : déduction max mensuelle (si eligible)
      - artistique irrégulier : déduction annuelle /12 (si eligible), non cumulable => à gérer par choix dans l'UI
      - ALE : on ajoute seulement la part excédentaire (simplification)
    """
    total = 0.0
    for r in revenus:
        m = max(0.0, float(r.get("montant_mensuel", 0.0)))
        t = r.get("type", "standard")
        eligible = bool(r.get("eligible", True))

        if t in ("socio_prof", "etudiant") and eligible:
            ded = min(float(cfg_soc["max_mensuel"]), m)
            total += max(0.0, m - ded)
        elif t == "artistique_irregulier" and eligible:
            ded_m = float(cfg_soc["artistique_annuel"]) / 12.0
            total += max(0.0, m - min(ded_m, m))
        elif t == "ale":
            # Simplifié: tu fournis directement la part à compter (celle > 4,10€)
            total += max(0.0, float(r.get("ale_part_excedentaire", 0.0)))
        else:
            total += m

    return total


# ============================================================
# COHABITANTS (art. 34 AR) + AVANTAGE EN NATURE (art. 33 AR)
# ============================================================
def cohabitant_part_a_compter(montant_cohab: float,
                              type_cohab: str,
                              categorie: str,
                              taux_ris_cohab: float,
                              pourcentage_facultatif: float) -> float:
    """
    type_cohab:
      - "aucun"
      - "conjoint_partenaire"
      - "asc_desc_1er_deg"
      - "autre"
    Règles:
      - conjoint/partenaire:
           * si categorie fam_charge: tout compter
           * sinon: compter la partie qui dépasse le taux cohabitant
      - asc/desc 1er degré: facultatif, sur la partie qui dépasse le taux cohabitant
      - autres: 0
    """
    m = max(0.0, float(montant_cohab))
    taux = max(0.0, float(taux_ris_cohab))
    pct = clamp01(pourcentage_facultatif)

    if type_cohab == "conjoint_partenaire":
        if categorie == "fam_charge":
            return m
        return max(0.0, m - taux)

    if type_cohab == "asc_desc_1er_deg":
        return pct * max(0.0, m - taux)

    return 0.0


# ============================================================
# IMMUNISATION SIMPLE (art. 22 §2 AR)
# ============================================================
def immunisation_simple_monthly(categorie: str, cfg_immu: dict) -> float:
    if categorie == "cohab":
        return float(cfg_immu["cohab"]) / 12.0
    if categorie == "fam_charge":
        return float(cfg_immu["fam_charge"]) / 12.0
    return float(cfg_immu["isole"]) / 12.0


# ============================================================
# CALCUL GLOBAL
# ============================================================
def compute_all(answers: dict, engine: dict) -> dict:
    cfg = engine["config"]

    # 1) revenus (avec exo socio-pro simplifiée)
    revenus = revenus_mensuels_total(answers.get("revenus", []), cfg["socio_prof"])

    # 2) capitaux mobiliers
    cap = capital_mobilier_monthly(
        total_capital=answers.get("capital_mobilier_total", 0.0),
        compte_commun=answers.get("capital_compte_commun", False),
        nb_titulaires=answers.get("capital_nb_titulaires", 1),
        categorie=answers.get("categorie", "isole"),
        conjoint_compte_commun=answers.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=answers.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    )

    # 3) biens immobiliers
    immo = immo_monthly_total(
        biens=answers.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )

    # 4) cession de biens
    cession = cession_biens_monthly(
        cessions=answers.get("cessions", []),
        categorie=answers.get("categorie", "isole"),
        conjoint_fraction_in_division=answers.get("cession_fam_charge_conjoint_indivision", False),
        fraction_demandeur=answers.get("cession_fraction_demandeur", 1.0),
        fraction_demandeur_et_conjoint=answers.get("cession_fraction_demandeur_et_conjoint", 1.0),
        cas_particulier_tranche_37200=answers.get("cession_cas_particulier_37200", False),
        dettes_deductibles=answers.get("cession_dettes_deductibles", 0.0),
        abatt_cat=answers.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=answers.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    )

    # 5) cohabitant (art.34)
    taux_cohab = float(cfg["ris_rates"]["cohab"])
    cohab = cohabitant_part_a_compter(
        montant_cohab=answers.get("cohabitant_montant", 0.0),
        type_cohab=answers.get("cohabitant_type", "aucun"),
        categorie=answers.get("categorie", "isole"),
        taux_ris_cohab=taux_cohab,
        pourcentage_facultatif=answers.get("cohabitant_pct", 0.0)
    )

    # 6) avantage nature (art.33)
    avantage_nature = max(0.0, float(answers.get("avantage_nature_logement", 0.0)))

    total_avant = revenus + cap + immo + cession + cohab + avantage_nature

    # taux RIS
    cat = answers.get("categorie", "isole")
    taux_ris = float(cfg["ris_rates"].get(cat, 0.0))

    # immunisation simple si ressources < taux
    immu = 0.0
    if taux_ris > 0 and total_avant < taux_ris:
        immu = immunisation_simple_monthly(cat, cfg["immunisation_simple_annuelle"])

    total_apres = max(0.0, total_avant - immu)
    ris = max(0.0, taux_ris - total_apres) if taux_ris > 0 else 0.0

    return {
        "revenus_mensuels_apres_exonerations": revenus,
        "capitaux_mobiliers_mensuels": cap,
        "immo_mensuels": immo,
        "cession_biens_mensuelle": cession,
        "cohabitant_part_a_compter": cohab,
        "avantage_nature_logement": avantage_nature,
        "total_ressources_avant_immunisation_simple": total_avant,
        "taux_ris_mensuel": taux_ris,
        "immunisation_simple_mensuelle": immu,
        "total_ressources_apres_immunisation_simple": total_apres,
        "ris_theorique": ris
    }


# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(page_title="Calcul RIS (prototype)", layout="centered")
st.title("Calcul RIS – Prototype (règles circulaire intégrées)")
st.caption("Objectif: encoder la situation → calculer ressources mensuelles et RIS théorique. (Prototype évolutif)")

engine = load_engine()
cfg = engine["config"]

# ---------------- Sidebar config ----------------
with st.sidebar:
    st.subheader("Paramètres (à compléter)")
    st.write("**Taux RIS mensuels officiels** (à encoder) :")
    cfg["ris_rates"]["cohab"] = st.number_input("Taux RIS cohabitant (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["cohab"]))
    cfg["ris_rates"]["isole"] = st.number_input("Taux RIS isolé (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["isole"]))
    cfg["ris_rates"]["fam_charge"] = st.number_input("Taux RIS famille à charge (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["fam_charge"]))

    st.divider()
    st.write("**Montants indexés (si besoin)**")
    cfg["socio_prof"]["max_mensuel"] = st.number_input("Exo socio-pro max (€/mois)", min_value=0.0, value=float(cfg["socio_prof"]["max_mensuel"]))
    cfg["socio_prof"]["artistique_annuel"] = st.number_input("Exo activité artistique irrégulière (€/an)", min_value=0.0, value=float(cfg["socio_prof"]["artistique_annuel"]))

    st.divider()
    st.write("**Immunisation simple (€/an)**")
    cfg["immunisation_simple_annuelle"]["cohab"] = st.number_input("Immunisation simple cohab (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["cohab"]))
    cfg["immunisation_simple_annuelle"]["isole"] = st.number_input("Immunisation simple isolé (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["isole"]))
    cfg["immunisation_simple_annuelle"]["fam_charge"] = st.number_input("Immunisation simple fam. charge (€/an)", min_value=0.0, value=float(cfg["immunisation_simple_annuelle"]["fam_charge"]))

# ---------------- Answers ----------------
answers = {}

answers["categorie"] = st.selectbox("Catégorie", ["cohab", "isole", "fam_charge"])
answers["enfants_a_charge"] = st.number_input("Enfants à charge", min_value=0, value=0, step=1)

# ---------------- Revenus ----------------
st.divider()
st.subheader("1) Revenus mensuels (nets) – avec exonérations socio-pro")
st.caption("Choisis le type de revenu si une exonération s’applique (socio-pro, étudiant, artistique irrégulier, ALE).")

revenus = []
nb_rev = st.number_input("Nombre de revenus à encoder", min_value=0, value=1, step=1)
for i in range(int(nb_rev)):
    st.markdown(f"**Revenu {i+1}**")
    c1, c2, c3 = st.columns([2, 1, 1])
    label = c1.text_input(f"Type/label (revenu {i+1})", value="salaire/chômage", key=f"rev_label_{i}")
    montant = c2.number_input(f"Montant net (€/mois) {i+1}", min_value=0.0, value=0.0, step=10.0, key=f"rev_m_{i}")
    typ = c3.selectbox(f"Règle {i+1}", ["standard", "socio_prof", "etudiant", "artistique_irregulier", "ale"], key=f"rev_t_{i}")

    eligible = True
    ale_part_exc = 0.0
    if typ in ("socio_prof", "etudiant", "artistique_irregulier"):
        eligible = st.checkbox(f"Éligible à l’exonération ? (revenu {i+1})", value=True, key=f"rev_el_{i}")
    if typ == "ale":
        ale_part_exc = st.number_input(f"Part ALE à compter (>4,10€) (€/mois) {i+1}", min_value=0.0, value=0.0, step=1.0, key=f"rev_ale_{i}")

    revenus.append({
        "label": label,
        "montant_mensuel": float(montant),
        "type": typ,
        "eligible": eligible,
        "ale_part_excedentaire": float(ale_part_exc)
    })

answers["revenus"] = revenus

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
        st.info("Rappel règle: fraction = 1/nb titulaires, ou 2/nb si 'famille à charge' + conjoint co-titulaire.")
    else:
        answers["capital_fraction"] = st.number_input("Part du demandeur (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

# ---------------- Biens immobiliers ----------------
st.divider()
st.subheader("3) Biens immobiliers (RC non indexé, (RC - exo) × 3)")
st.caption("Inclut multipropriété (exo divisé par nb biens), indivision (RC & exo × fraction), hypothèque/viager (plafond 50%).")

biens = []
a_immo = st.checkbox("Le demandeur possède des biens immobiliers")
if a_immo:
    nb_biens = st.number_input("Nombre de biens à encoder", min_value=0, value=1, step=1)
    for i in range(int(nb_biens)):
        st.markdown(f"**Bien {i+1}**")
        label = st.text_input(f"Libellé (bien {i+1})", value="terrain/maison", key=f"im_label_{i}")
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
            "label": label,
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
st.caption("Valeur vénale → calcul comme capitaux mobiliers (même si produit plus disponible). Tranche 37.200€ (cas particuliers) + dettes + abattement proratisé.")

cessions = []
a_ces = st.checkbox("Le demandeur a cédé des biens dans les 10 dernières années")
answers["cessions"] = []
answers["cession_fraction_demandeur"] = 1.0
answers["cession_fraction_demandeur_et_conjoint"] = 1.0
answers["cession_fam_charge_conjoint_indivision"] = False
answers["cession_cas_particulier_37200"] = False
answers["cession_dettes_deductibles"] = 0.0
answers["cession_abatt_cat"] = "cat1"
answers["cession_abatt_mois"] = 0

if a_ces:
    st.markdown("**Pondérations de droits (si indivision)**")
    answers["cession_fraction_demandeur"] = st.number_input("Fraction globale du demandeur (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
    if answers["categorie"] == "fam_charge":
        answers["cession_fam_charge_conjoint_indivision"] = st.checkbox(
            "Cession en indivision avec conjoint/partenaire (famille à charge) → appliquer fraction demandeur+conjoint",
            value=False
        )
        if answers["cession_fam_charge_conjoint_indivision"]:
            answers["cession_fraction_demandeur_et_conjoint"] = st.number_input("Fraction demandeur + conjoint (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)

    answers["cession_cas_particulier_37200"] = st.checkbox("Cas particulier: tranche immunisée 37.200€ applicable", value=False)

    st.markdown("**Dettes personnelles déductibles (si conditions remplies)**")
    dettes_ok = st.checkbox("Conditions OK (dettes perso, contractées avant, payées avec produit) ?", value=False)
    if dettes_ok:
        answers["cession_dettes_deductibles"] = st.number_input("Montant total des dettes déductibles (€)", min_value=0.0, value=0.0, step=100.0)

    st.markdown("**Abattement annuel proratisé (cat. 1/2/3)**")
    answers["cession_abatt_cat"] = st.selectbox("Catégorie d’abattement", ["cat1", "cat2", "cat3"])
    answers["cession_abatt_mois"] = st.number_input("Nombre de mois (prorata)", min_value=0, max_value=12, value=0, step=1)

    nb_c = st.number_input("Nombre de cessions à encoder", min_value=0, value=1, step=1)
    for i in range(int(nb_c)):
        st.markdown(f"**Cession {i+1}**")
        val = st.number_input(f"Valeur vénale (€) (cession {i+1})", min_value=0.0, value=0.0, step=100.0, key=f"ces_v_{i}")
        usuf = st.checkbox(f"Cession d’usufruit ? (cession {i+1})", value=False, key=f"ces_u_{i}")
        indiv = st.checkbox(f"En indivision ? (cession {i+1})", value=False, key=f"ces_i_{i}")
        frac = 1.0
        if indiv:
            frac = st.number_input(f"Fraction de droits (0–1) (cession {i+1})", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"ces_f_{i}")

        cessions.append({
            "valeur_venale": float(val),
            "usufruit": usuf,
            "en_indivision": indiv,
            "fraction_droits": float(frac)
        })

    answers["cessions"] = cessions

# ---------------- Cohabitation ----------------
st.divider()
st.subheader("5) Cohabitation (art. 34 AR)")
type_cohab = st.selectbox("Type de cohabitation", ["aucun", "conjoint_partenaire", "asc_desc_1er_deg", "autre"])
answers["cohabitant_type"] = type_cohab
answers["cohabitant_montant"] = 0.0
answers["cohabitant_pct"] = 0.0

if type_cohab != "aucun":
    answers["cohabitant_montant"] = st.number_input("Ressources mensuelles nettes du cohabitant (€/mois)", min_value=0.0, value=0.0, step=10.0)
    if type_cohab == "asc_desc_1er_deg":
        answers["cohabitant_pct"] = st.slider("Pourcentage pris en compte (facultatif)", min_value=0, max_value=100, value=100, step=5) / 100.0

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

    st.success("Calcul terminé ✅")
    st.metric("RIS théorique (€/mois)", f"{res['ris_theorique']:.2f}")

    st.write("### Détail (mensuel)")
    st.json(res)

    st.info("💡 Si tu veux sauvegarder les paramètres dans ris_rules.json, je peux te générer le fichier aussi.")

