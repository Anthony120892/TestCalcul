import json
import streamlit as st
from types import SimpleNamespace

# ---------- Moteur (repris de ton runner) ----------
def capital_mobilier_monthly(total, fraction, cap_cfg):
    total = max(0.0, float(total)) * max(0.0, min(1.0, float(fraction)))
    t0 = cap_cfg["t0_max"]; t1_min = cap_cfg["t1_min"]; t1_max = cap_cfg["t1_max"]
    r1 = cap_cfg["t1_rate"]; r2 = cap_cfg["t2_rate"]
    annuel = 0.0
    if total <= t0:
        annuel = 0.0
    else:
        tranche1 = max(0.0, min(total, t1_max) - t1_min)
        annuel += tranche1 * r1
        tranche2 = max(0.0, total - t1_max)
        annuel += tranche2 * r2
    return annuel / 12.0

def immo_one_monthly(p, enfants, exo_cfg):
    if p.get("habitation_principale", False):
        return 0.0
    bati = bool(p.get("bati", True))
    rc = float(p.get("rc_indexe", 0.0))
    fraction = max(0.0, min(1.0, float(p.get("fraction", 1.0))))

    if bati:
        exo = exo_cfg["bati_base"] + exo_cfg["bati_par_enfant"] * max(0, int(enfants))
    else:
        exo = exo_cfg["non_bati_base"]

    base_annuelle = max(0.0, (rc - exo) * 3.0) * fraction

    if p.get("hypotheque", False):
        interets = float(p.get("interets_annuels", 0.0))
        if interets > 0:
            reduction = min(interets, 0.5 * base_annuelle)
            base_annuelle = max(0.0, base_annuelle - reduction)

    return base_annuelle / 12.0

def immo_monthly_total(biens, enfants, exo_cfg):
    return sum(immo_one_monthly(p, enfants, exo_cfg) for p in biens)

def immun_simple_monthly(categorie, immun_cfg):
    if categorie == "fam_charge":
        return float(immun_cfg["fam_charge"]) / 12.0
    return float(immun_cfg["cohab_ou_isole"]) / 12.0

def compute(answers, engine):
    cfg = engine["config"]
    derived = {}

    revenus = sum(r["montant_mensuel"] for r in answers.get("revenus", []))
    cap = capital_mobilier_monthly(
        answers.get("capital_mobilier_total", 0),
        answers.get("capital_mobilier_fraction", 1),
        cfg["capital_mobilier"]
    )
    immo = immo_monthly_total(
        answers.get("biens_immobiliers", []),
        answers.get("enfants_a_charge", 0),
        cfg["exonerations"]
    )
    cohab = answers.get("ressources_cohabitant_mensuelles", 0) if answers.get("inclure_ressources_cohabitant", False) else 0

    total = revenus + cap + immo + cohab
    taux = cfg["ris_rates"][answers.get("categorie", "isole")]

    immun = 0
    if taux > 0 and total < taux:
        immun = immun_simple_monthly(answers.get("categorie", "isole"), cfg["immunisation_simple_annuelle"])

    total_apres = max(0, total - immun)
    ris = max(0, taux - total_apres) if taux > 0 else 0

    derived.update({
        "revenus_mensuels_nets": revenus,
        "capitaux_mensuels": cap,
        "immo_mensuels": immo,
        "cohab_mensuels": cohab,
        "total_ressources_avant_immun": total,
        "taux_ris_mensuel": taux,
        "immunisation_simple_mensuelle": immun,
        "total_ressources_apres_immun": total_apres,
        "ris_theorique": ris
    })
    return derived

# ---------- UI Streamlit ----------
st.set_page_config(page_title="Calcul RIS", layout="centered")
st.title("Calcul RIS – Prototype (rubriques → calcul)")
st.caption("Remplis les rubriques, le moteur calcule les ressources mensuelles et le RIS théorique.")

engine = json.load(open("ris_rules.json", "r", encoding="utf-8"))

with st.sidebar:
    st.subheader("Paramètres (à mettre à jour)")
    st.write("Renseigne les taux RIS mensuels officiels :")
    engine["config"]["ris_rates"]["cohab"] = st.number_input("Taux RIS cohabitant (€/mois)", min_value=0.0, value=float(engine["config"]["ris_rates"]["cohab"]))
    engine["config"]["ris_rates"]["isole"] = st.number_input("Taux RIS isolé (€/mois)", min_value=0.0, value=float(engine["config"]["ris_rates"]["isole"]))
    engine["config"]["ris_rates"]["fam_charge"] = st.number_input("Taux RIS famille à charge (€/mois)", min_value=0.0, value=float(engine["config"]["ris_rates"]["fam_charge"]))

answers = {}

answers["categorie"] = st.selectbox("Catégorie", ["cohab", "isole", "fam_charge"])
answers["enfants_a_charge"] = st.number_input("Enfants à charge", min_value=0, value=0, step=1)

st.divider()
st.subheader("Revenus mensuels (nets)")
revenus = []
nb_rev = st.number_input("Nombre de revenus à encoder", min_value=0, value=1, step=1)
for i in range(int(nb_rev)):
    c1, c2 = st.columns(2)
    label = c1.text_input(f"Revenu {i+1} – type", value="salaire/chômage")
    montant = c2.number_input(f"Revenu {i+1} – montant net (€/mois)", min_value=0.0, value=0.0, step=10.0)
    revenus.append({"label": label, "montant_mensuel": float(montant)})
answers["revenus"] = revenus

st.divider()
st.subheader("Capitaux mobiliers")
a_cap = st.checkbox("Le demandeur possède des capitaux mobiliers")
if a_cap:
    answers["capital_mobilier_total"] = st.number_input("Montant total capitaux (€)", min_value=0.0, value=0.0, step=100.0)
    answers["capital_mobilier_fraction"] = st.number_input("Part du demandeur (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
else:
    answers["capital_mobilier_total"] = 0.0
    answers["capital_mobilier_fraction"] = 1.0

st.divider()
st.subheader("Biens immobiliers")
biens = []
a_immo = st.checkbox("Le demandeur possède des biens immobiliers")
if a_immo:
    nb_biens = st.number_input("Nombre de biens à encoder", min_value=0, value=1, step=1)
    for i in range(int(nb_biens)):
        st.markdown(f"**Bien {i+1}**")
        label = st.text_input(f"Libellé (bien {i+1})", value="terrain/maison")
        habitation_principale = st.checkbox(f"Habitation principale ? (bien {i+1})", value=False)
        bati = st.checkbox(f"Bien bâti ? (bien {i+1})", value=True)
        rc_indexe = st.number_input(f"RC indexé annuel (bien {i+1})", min_value=0.0, value=0.0, step=100.0)
        fraction = st.number_input(f"Fraction (0–1) (bien {i+1})", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
        hypotheque = False
        interets_annuels = 0.0
        if not habitation_principale:
            hypotheque = st.checkbox(f"Hypothèque ? (bien {i+1})", value=False)
            if hypotheque:
                interets_annuels = st.number_input(f"Intérêts annuels (bien {i+1})", min_value=0.0, value=0.0, step=100.0)

        biens.append({
            "label": label,
            "habitation_principale": habitation_principale,
            "bati": bati,
            "rc_indexe": float(rc_indexe),
            "fraction": float(fraction),
            "hypotheque": hypotheque,
            "interets_annuels": float(interets_annuels)
        })
answers["biens_immobiliers"] = biens

st.divider()
st.subheader("Cohabitation")
cohab = st.checkbox("Le demandeur est en cohabitation")
answers["cohabitation"] = cohab
answers["inclure_ressources_cohabitant"] = False
answers["ressources_cohabitant_mensuelles"] = 0.0
if cohab:
    incl = st.checkbox("Inclure les ressources du cohabitant")
    answers["inclure_ressources_cohabitant"] = incl
    if incl:
        answers["ressources_cohabitant_mensuelles"] = st.number_input("Ressources cohabitant (€/mois)", min_value=0.0, value=0.0, step=10.0)

st.divider()
if st.button("Calculer le RIS"):
    res = compute(answers, engine)

    st.success("Calcul terminé")
    st.metric("RIS théorique (€/mois)", f"{res['ris_theorique']:.2f}")
    st.write("Détail des ressources (mensuel) :")
    st.json(res)
