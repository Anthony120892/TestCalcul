import json
import os
import streamlit as st

# ============================================================
# CONFIG PAR DÉFAUT (complet) — on fusionne avec ris_rules.json
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

        # Exonérations socio-pro (immunisation double) — paramètres “indexables”
        "socio_prof": {
            "max_mensuel": 274.82,          # exo socio-pro max / mois
            "artistique_annuel": 3297.80,   # exo artistique irrégulier / an
        },

        # Cession de biens (art. 28 à 32 AR) — version “moteur”
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
    Rend ton ris_rules.json minimal compatible avec le moteur complet.
    - gère les anciens noms (exonerations -> immo)
    - complète les sections manquantes
    """
    raw = raw or {}
    engine = deep_merge(DEFAULT_ENGINE, raw)
    cfg = engine["config"]

    # Si l’utilisateur n’a que "exonerations", on mappe vers "immo"
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
    if os.path.exists("ris_rules.json"):
        with open("ris_rules.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        return normalize_engine(raw)
    return normalize_engine(DEFAULT_ENGINE)

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
    Règle (annuelle):
      - 0% jusqu’à 6.199
      - 6% entre 6.200 et 12.500
      - 10% au-delà

    Compte commun:
      - fraction = 1/nb titulaires
      - si categorie == fam_charge ET conjoint co-titulaire => fraction = 2/nb
    Sinon:
      - fraction = part_fraction_custom
    """
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
# IMMOBILIER (RC non indexé) — avec multipropriété + indivision + hypo/viager (plafond 50%)
# ============================================================
def immo_monthly_total(biens: list, enfants: int, cfg_immo: dict) -> float:
    biens_countes = [b for b in biens if not b.get("habitation_principale", False)]

    nb_bati = sum(1 for b in biens_countes if b.get("bati", True))
    nb_non_bati = sum(1 for b in biens_countes if not b.get("bati", True))
    nb_bati = nb_bati if nb_bati > 0 else 0
    nb_non_bati = nb_non_bati if nb_non_bati > 0 else 0

    exo_bati_total = float(cfg_immo["bati_base"]) + float(cfg_immo["bati_par_enfant"]) * max(0, int(enfants))
    exo_non_bati_total = float(cfg_immo["non_bati_base"])
    coeff = float(cfg_immo.get("coeff_rc", 3.0))

    total_annuel = 0.0

    for b in biens_countes:
        bati = bool(b.get("bati", True))
        rc = max(0.0, float(b.get("rc_non_indexe", 0.0)))
        frac = clamp01(b.get("fraction_droits", 1.0))

        # Indivision: RC * fraction avant calcul
        rc_part = rc * frac

        # Multipropriété: exo divisé par nb de biens du type
        if bati:
            if nb_bati == 0:
                exo_par_bien = 0.0
            else:
                exo_par_bien = (exo_bati_total * frac) / nb_bati
        else:
            if nb_non_bati == 0:
                exo_par_bien = 0.0
            else:
                exo_par_bien = (exo_non_bati_total * frac) / nb_non_bati

        base = max(0.0, (rc_part - exo_par_bien) * coeff)

        # Hypothèque: intérêts * fraction, plafonné à 50% du montant à prendre en compte
        if b.get("hypotheque", False):
            interets = max(0.0, float(b.get("interets_annuels", 0.0))) * frac
            base -= min(interets, 0.5 * base)

        # Viager: rente * fraction, plafonné à 50%
        if b.get("viager", False):
            rente = max(0.0, float(b.get("rente_viagere_annuelle", 0.0))) * frac
            base -= min(rente, 0.5 * base)

        total_annuel += max(0.0, base)

    return total_annuel / 12.0

# ============================================================
# CESSION DE BIENS (art. 28 à 32 AR) — version “moteur”
# ============================================================
def cession_biens_monthly(cessions: list,
                          categorie: str,
                          cas_particulier_tranche_37200: bool,
                          dettes_deductibles: float,
                          abatt_cat: str,
                          abatt_mois_prorata: int,
                          cfg_cession: dict,
                          cfg_cap: dict) -> float:
    """
    - on totalise les valeurs (si plusieurs cessions: tranches 1 seule fois sur total)
    - usufruit => 40% (si coché)
    - dettes déductibles => option manuelle (si conditions remplies)
    - tranche immunisée 37.200 => option manuelle (cas particulier)
    - abattement annuel proratisé
    - puis calcul par tranches comme capitaux mobiliers
    """
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

    # Calcul comme capitaux mobiliers (seuils non fractionnés ici)
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
# REVENUS (exo socio-pro simplifiée)
# ============================================================
def revenus_mensuels_total(revenus: list, cfg_soc: dict) -> float:
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
            total += max(0.0, float(r.get("ale_part_excedentaire", 0.0)))
        else:
            total += m
    return total

# ============================================================
# IMMUNISATION SIMPLE (art. 22 §2 AR) — mensuel
# ============================================================
def immunisation_simple_monthly(categorie: str, cfg_immu: dict) -> float:
    return float(cfg_immu.get(categorie, 0.0)) / 12.0

# ============================================================
# COHABITANT — version calée sur les décisions CPAS que tu as fournies
# ============================================================
def cohabitant_monthly_cpasmode(cohabitant_revenus_annuels: float,
                                categorie_demandeur: str,
                                taux_annuel_reference: float,
                                immun_simple_annuelle_cohab: float,
                                appliquer_division_par_2: bool = True) -> float:
    """
    Logique observée dans la décision:
      (revenus_cohabitant_annuels - taux_reference_annuel) / 2 - immunisation_simple_cohab
    où taux_reference_annuel = "cat.E" (dans tes annexes: 21.312,87/an)
    """
    r = max(0.0, float(cohabitant_revenus_annuels))
    ref = max(0.0, float(taux_annuel_reference))
    immu = max(0.0, float(immun_simple_annuelle_cohab))

    base = max(0.0, r - ref)
    if appliquer_division_par_2:
        base = base / 2.0
    base = max(0.0, base - immu)
    return base / 12.0

# ============================================================
# CALCUL GLOBAL
# ============================================================
def compute_all(answers: dict, engine: dict) -> dict:
    cfg = engine["config"]

    cat = answers.get("categorie", "isole")
    taux_ris_m = float(cfg["ris_rates"].get(cat, 0.0))
    taux_ris_annuel = taux_ris_m * 12.0

    # 1) revenus demandeur
    revenus = revenus_mensuels_total(answers.get("revenus", []), cfg["socio_prof"])

    # 2) capitaux mobiliers
    cap = capital_mobilier_monthly(
        total_capital=answers.get("capital_mobilier_total", 0.0),
        compte_commun=answers.get("capital_compte_commun", False),
        nb_titulaires=answers.get("capital_nb_titulaires", 1),
        categorie=cat,
        conjoint_compte_commun=answers.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=answers.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    )

    # 3) immobilier
    immo = immo_monthly_total(
        biens=answers.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )

    # 4) cession
    cession = cession_biens_monthly(
        cessions=answers.get("cessions", []),
        categorie=cat,
        cas_particulier_tranche_37200=answers.get("cession_cas_particulier_37200", False),
        dettes_deductibles=answers.get("cession_dettes_deductibles", 0.0),
        abatt_cat=answers.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=answers.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    )

    # 5) cohabitant — MODE “CPAS décision”
    cohab_mode = answers.get("cohab_mode", "cpas")
    cohab_m = 0.0
    if cohab_mode == "cpas":
        # Ici tu encodes le cohabitant en ANNUEL, comme dans les décisions
        revenus_cohab_ann = float(answers.get("cohabitant_annuel_total", 0.0))

        # Le "taux de référence" dans tes décisions correspond à 21.312,87/an (cat.E),
        # c’est le RIS "personne avec charge de famille" annuel.
        # On le prend depuis le taux fam_charge (mensuel) si tu l’as encodé,
        # sinon tu peux forcer un champ manuel.
        ref_annuel_auto = float(cfg["ris_rates"].get("fam_charge", 0.0)) * 12.0
        ref_annuel = float(answers.get("cohabitant_ref_annuel", ref_annuel_auto))

        immu_cohab_ann = float(cfg["immunisation_simple_annuelle"].get("cohab", 155.0))
        div2 = bool(answers.get("cohabitant_div2", True))

        cohab_m = cohabitant_monthly_cpasmode(
            cohabitant_revenus_annuels=revenus_cohab_ann,
            categorie_demandeur=cat,
            taux_annuel_reference=ref_annuel,
            immun_simple_annuelle_cohab=immu_cohab_ann,
            appliquer_division_par_2=div2
        )

    # 6) avantage nature
    avantage_nature = max(0.0, float(answers.get("avantage_nature_logement", 0.0)))

    total_avant = revenus + cap + immo + cession + cohab_m + avantage_nature

    # immunisation simple si ressources < taux
    immu_m = 0.0
    if taux_ris_m > 0 and total_avant < taux_ris_m:
        immu_m = immunisation_simple_monthly(cat, cfg["immunisation_simple_annuelle"])

    total_apres = max(0.0, total_avant - immu_m)
    ris = max(0.0, taux_ris_m - total_apres) if taux_ris_m > 0 else 0.0

    return {
        "revenus_mensuels_apres_exonerations": revenus,
        "capitaux_mobiliers_mensuels": cap,
        "immo_mensuels": immo,
        "cession_biens_mensuelle": cession,
        "cohabitant_part_a_compter_mensuel": cohab_m,
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
st.title("Calcul RIS – Prototype (moteur robuste + mode CPAS)")
st.caption("Tu encodes → le moteur calcule. Et surtout: il ne crashe plus si le JSON est minimal 😄")

engine = load_engine()
cfg = engine["config"]

# ---------------- Sidebar config ----------------
with st.sidebar:
    st.subheader("Paramètres")
    st.write("**Taux RIS mensuels officiels** :")
    cfg["ris_rates"]["cohab"] = st.number_input("Taux RIS cohabitant (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["cohab"]))
    cfg["ris_rates"]["isole"] = st.number_input("Taux RIS isolé (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["isole"]))
    cfg["ris_rates"]["fam_charge"] = st.number_input("Taux RIS famille à charge (€/mois)", min_value=0.0, value=float(cfg["ris_rates"]["fam_charge"]))

    st.divider()
    st.write("**Montants indexables**")
    cfg["socio_prof"]["max_mensuel"] = st.number_input("Exo socio-pro max (€/mois)", min_value=0.0, value=float(cfg["socio_prof"]["max_mensuel"]))
    cfg["socio_prof"]["artistique_annuel"] = st.number_input("Exo artistique irrégulier (€/an)", min_value=0.0, value=float(cfg["socio_prof"]["artistique_annuel"]))

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

# ---------------- Cohabitant (mode CPAS) ----------------
st.divider()
st.subheader("5) Ressources du cohabitant — mode CPAS (comme tes décisions)")

answers["cohab_mode"] = "cpas"

answers["cohabitant_annuel_total"] = st.number_input(
    "Total annuel des ressources du cohabitant (€/an) (ex: salaire+mutuelle)",
    min_value=0.0, value=0.0, step=100.0
)

# Référence annuelle (souvent = RIS famille à charge annuel, ex 21.312,87/an)
ref_auto = float(cfg["ris_rates"]["fam_charge"]) * 12.0
answers["cohabitant_ref_annuel"] = st.number_input(
    "Référence annuelle à déduire (€/an) (souvent = RIS 'famille à charge' annuel)",
    min_value=0.0, value=float(ref_auto), step=100.0
)

answers["cohabitant_div2"] = st.checkbox("Diviser par 2 (ménage à 2)", value=True)

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
