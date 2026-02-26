Test 

import json
import os
import calendar
from datetime import date, timedelta
from io import BytesIO

import streamlit as st

# ============================================================
# CONFIG PAR DÉFAUT (fusion avec ris_rules.json si présent)
# ============================================================
DEFAULT_ENGINE = {
    "version": "1.7",  # ✅ bump interne
    "config": {
        "ris_rates_annuel": {"cohab": 10513.60, "isole": 15770.41, "fam_charge": 21312.87},
        "ris_rates": {"cohab": 876.13, "isole": 1314.20, "fam_charge": 1776.07},
        "immunisation_simple_annuelle": {"cohab": 155.0, "isole": 250.0, "fam_charge": 310.0},
        "art34": {"taux_a_laisser_mensuel": 876.13},
        "pf": {"pf_mensuel_defaut": 240.0},
        "capital_mobilier": {
            "t0_max": 6199.0,
            "t1_min": 6200.0,
            "t1_max": 12500.0,
            "t1_rate": 0.06,
            "t2_rate": 0.10
        },
        "immo": {
            "bati_base": 750.0,
            "bati_par_enfant": 125.0,
            "non_bati_base": 30.0,
            "coeff_rc": 3.0
        },
        "socio_prof": {
            "max_mensuel": 309.48,
            "artistique_annuel": 3297.80,
        },
        "cession": {
            "tranche_immunisee": 37200.0,
            "usufruit_ratio": 0.40,
            "abattements_annuels": {"cat1": 1250.0, "cat2": 2000.0, "cat3": 2500.0}
        },
        "ale": {"valeur_cheque": 4.35, "exon_par_cheque": 6.0}
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

    if "exonerations" in cfg and "immo" in cfg:
        exo = cfg["exonerations"]
        cfg["immo"]["bati_base"] = float(exo.get("bati_base", cfg["immo"]["bati_base"]))
        cfg["immo"]["bati_par_enfant"] = float(exo.get("bati_par_enfant", cfg["immo"]["bati_par_enfant"]))
        cfg["immo"]["non_bati_base"] = float(exo.get("non_bati_base", cfg["immo"]["non_bati_base"]))

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

    if "pf" not in cfg:
        cfg["pf"] = {"pf_mensuel_defaut": 0.0}
    cfg["pf"]["pf_mensuel_defaut"] = float(cfg["pf"].get("pf_mensuel_defaut", 0.0))

    if "ale" not in cfg:
        cfg["ale"] = {"valeur_cheque": 0.0, "exon_par_cheque": 6.0}
    cfg["ale"]["valeur_cheque"] = float(cfg["ale"].get("valeur_cheque", 0.0))
    cfg["ale"]["exon_par_cheque"] = float(cfg["ale"].get("exon_par_cheque", 6.0))

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

def safe_parse_date(x):
    if isinstance(x, date):
        return x
    if isinstance(x, str) and x.strip():
        try:
            return date.fromisoformat(x.strip())
        except Exception:
            return None
    return None

# ============================================================
# CAPITAUX MOBILIERS (annuel) - détail tranches
# ============================================================
def capital_mobilier_calc(total_capital: float,
                          compte_commun: bool,
                          nb_titulaires: int,
                          categorie: str,
                          conjoint_compte_commun: bool,
                          part_fraction_custom: float,
                          cfg_cap: dict) -> dict:
    total_capital = max(0.0, float(total_capital))

    if compte_commun:
        nb = max(1, int(nb_titulaires))
        numerator = 2 if (categorie == "fam_charge" and conjoint_compte_commun) else 1
        fraction = numerator / nb
        fraction_mode = f"compte commun: {numerator}/{nb}"
    else:
        fraction = clamp01(part_fraction_custom)
        fraction_mode = f"fraction custom: {fraction:.2f}"

    adj_total = total_capital * fraction

    t0_max = float(cfg_cap["t0_max"]) * fraction
    t1_min = float(cfg_cap["t1_min"]) * fraction
    t1_max = float(cfg_cap["t1_max"]) * fraction
    r1_ = float(cfg_cap["t1_rate"])
    r2_ = float(cfg_cap["t2_rate"])

    tranche1_base = max(0.0, min(adj_total, t1_max) - t1_min)
    tranche2_base = max(0.0, adj_total - t1_max)

    tranche1_calc = tranche1_base * r1_
    tranche2_calc = tranche2_base * r2_

    if adj_total <= t0_max:
        annuel = 0.0
    else:
        annuel = tranche1_calc + tranche2_calc

    return {
        "total_capital": r2(total_capital),
        "fraction": r2(fraction),
        "fraction_mode": fraction_mode,
        "capital_pris_en_compte_base": r2(adj_total),
        "seuils": {
            "t0_max": r2(t0_max),
            "t1_min": r2(t1_min),
            "t1_max": r2(t1_max),
            "t1_rate": r2(r1_),
            "t2_rate": r2(r2_),
        },
        "tranches": [
            {"label": "Tranche 1", "base": r2(tranche1_base), "taux": r2(r1_), "produit": r2(tranche1_calc),
             "borne": f"]{r2(t1_min)} ; {r2(t1_max)}]"},
            {"label": "Tranche 2", "base": r2(tranche2_base), "taux": r2(r2_), "produit": r2(tranche2_calc),
             "borne": f">{r2(t1_max)}"},
        ],
        "annuel": r2(max(0.0, annuel))
    }

def capital_mobilier_annuel(total_capital: float,
                            compte_commun: bool,
                            nb_titulaires: int,
                            categorie: str,
                            conjoint_compte_commun: bool,
                            part_fraction_custom: float,
                            cfg_cap: dict) -> float:
    return float(capital_mobilier_calc(
        total_capital=total_capital,
        compte_commun=compte_commun,
        nb_titulaires=nb_titulaires,
        categorie=categorie,
        conjoint_compte_commun=conjoint_compte_commun,
        part_fraction_custom=part_fraction_custom,
        cfg_cap=cfg_cap
    )["annuel"])

# ============================================================
# IMMOBILIER (annuel) - détail
# ============================================================
def immo_calc_total(biens: list, enfants: int, cfg_immo: dict) -> dict:
    biens_countes = [b for b in biens if not b.get("habitation_principale", False)]
    nb_bati = sum(1 for b in biens_countes if b.get("bati", True))
    nb_non_bati = sum(1 for b in biens_countes if not b.get("bati", True))

    exo_bati_total = float(cfg_immo["bati_base"]) + float(cfg_immo["bati_par_enfant"]) * max(0, int(enfants))
    exo_non_bati_total = float(cfg_immo["non_bati_base"])
    coeff = float(cfg_immo.get("coeff_rc", 3.0))

    details = []
    total_annuel = 0.0

    for idx, b in enumerate(biens_countes, start=1):
        bati = bool(b.get("bati", True))
        rc = max(0.0, float(b.get("rc_non_indexe", 0.0)))
        frac = clamp01(b.get("fraction_droits", 1.0))
        rc_part = rc * frac

        if bati:
            exo_par_bien = (exo_bati_total * frac) / nb_bati if nb_bati > 0 else 0.0
            typ = "Bâti"
        else:
            exo_par_bien = (exo_non_bati_total * frac) / nb_non_bati if nb_non_bati > 0 else 0.0
            typ = "Non bâti"

        base_rc = max(0.0, rc_part - exo_par_bien)
        base_coeff = base_rc * coeff

        ded_interets = 0.0
        ded_rente = 0.0

        if b.get("hypotheque", False):
            interets = max(0.0, float(b.get("interets_annuels", 0.0))) * frac
            ded_interets = min(interets, 0.5 * base_coeff)
            base_coeff -= ded_interets

        if b.get("viager", False):
            rente = max(0.0, float(b.get("rente_viagere_annuelle", 0.0))) * frac
            ded_rente = min(rente, 0.5 * base_coeff)
            base_coeff -= ded_rente

        pris = max(0.0, base_coeff)
        total_annuel += pris

        details.append({
            "bien": idx,
            "type": typ,
            "rc_non_indexe": r2(rc),
            "fraction": r2(frac),
            "rc_part": r2(rc_part),
            "exo_par_bien": r2(exo_par_bien),
            "rc_apres_exo": r2(base_rc),
            "coeff_rc": r2(coeff),
            "base_coeff": r2(base_rc * coeff),
            "ded_interets": r2(ded_interets),
            "ded_rente": r2(ded_rente),
            "pris_en_compte": r2(pris),
        })

    return {
        "total_annuel": r2(max(0.0, total_annuel)),
        "coeff_rc": r2(coeff),
        "exo_bati_total": r2(exo_bati_total),
        "exo_non_bati_total": r2(exo_non_bati_total),
        "nb_bati": int(nb_bati),
        "nb_non_bati": int(nb_non_bati),
        "details": details,
    }

def immo_annuel_total(biens: list, enfants: int, cfg_immo: dict) -> float:
    return float(immo_calc_total(biens, enfants, cfg_immo)["total_annuel"])

# ============================================================
# CESSION DE BIENS (annuel) - détail
# ============================================================
def cession_biens_calc(cessions: list,
                       cas_particulier_tranche_37200: bool,
                       dettes_deductibles: float,
                       abatt_cat: str,
                       abatt_mois_prorata: int,
                       cfg_cession: dict,
                       cfg_cap: dict) -> dict:
    brut = 0.0
    details_cess = []
    for i, c in enumerate(cessions or [], start=1):
        v = max(0.0, float(c.get("valeur_venale", 0.0)))
        v0 = v
        if c.get("usufruit", False):
            v = v * float(cfg_cession["usufruit_ratio"])
        brut += v
        details_cess.append({
            "cession": i,
            "valeur_venale": r2(v0),
            "usufruit": bool(c.get("usufruit", False)),
            "ratio_usufruit": r2(float(cfg_cession["usufruit_ratio"])),
            "valeur_retendue": r2(v),
        })

    dettes = max(0.0, float(dettes_deductibles))
    apres_dettes = max(0.0, brut - dettes)

    tranche_immunisee = float(cfg_cession["tranche_immunisee"]) if cas_particulier_tranche_37200 else 0.0
    apres_tranche = max(0.0, apres_dettes - tranche_immunisee)

    abatt_annuel = float(cfg_cession["abattements_annuels"].get(abatt_cat, 0.0))
    mois = max(0, min(12, int(abatt_mois_prorata)))
    abatt_prorata = abatt_annuel * (mois / 12.0)
    base_cap = max(0.0, apres_tranche - abatt_prorata)

    t0_max = float(cfg_cap["t0_max"])
    t1_min = float(cfg_cap["t1_min"])
    t1_max = float(cfg_cap["t1_max"])
    r1_ = float(cfg_cap["t1_rate"])
    r2_ = float(cfg_cap["t2_rate"])

    tranche1_base = max(0.0, min(base_cap, t1_max) - t1_min)
    tranche2_base = max(0.0, base_cap - t1_max)
    tranche1_calc = tranche1_base * r1_
    tranche2_calc = tranche2_base * r2_

    if base_cap <= t0_max:
        annuel = 0.0
    else:
        annuel = tranche1_calc + tranche2_calc

    return {
        "brut_total": r2(brut),
        "details_cessions": details_cess,
        "dettes_deductibles": r2(dettes),
        "apres_dettes": r2(apres_dettes),
        "cas_tranche_37200": bool(cas_particulier_tranche_37200),
        "tranche_37200": r2(tranche_immunisee),
        "apres_tranche_37200": r2(apres_tranche),
        "abatt_cat": abatt_cat,
        "abatt_annuel": r2(abatt_annuel),
        "abatt_mois": int(mois),
        "abatt_prorata": r2(abatt_prorata),
        "base_cap": r2(base_cap),
        "seuils": {"t0_max": r2(t0_max), "t1_min": r2(t1_min), "t1_max": r2(t1_max), "t1_rate": r2(r1_), "t2_rate": r2(r2_)},
        "tranches": [
            {"label": "Tranche 1", "base": r2(tranche1_base), "taux": r2(r1_), "produit": r2(tranche1_calc), "borne": f"]{r2(t1_min)} ; {r2(t1_max)}]"},
            {"label": "Tranche 2", "base": r2(tranche2_base), "taux": r2(r2_), "produit": r2(tranche2_calc), "borne": f">{r2(t1_max)}"},
        ],
        "annuel": r2(max(0.0, annuel)),
    }

def cession_biens_annuelle(cessions: list,
                           cas_particulier_tranche_37200: bool,
                           dettes_deductibles: float,
                           abatt_cat: str,
                           abatt_mois_prorata: int,
                           cfg_cession: dict,
                           cfg_cap: dict) -> float:
    return float(cession_biens_calc(
        cessions=cessions,
        cas_particulier_tranche_37200=cas_particulier_tranche_37200,
        dettes_deductibles=dettes_deductibles,
        abatt_cat=abatt_cat,
        abatt_mois_prorata=abatt_mois_prorata,
        cfg_cession=cfg_cession,
        cfg_cap=cfg_cap
    )["annuel"])

# ============================================================
# REVENUS + ALE
# ============================================================
def _ale_montants(nb_cheques_mois: float, cfg_ale: dict) -> tuple[float, float, float]:
    nb = max(0.0, float(nb_cheques_mois))
    val = max(0.0, float(cfg_ale.get("valeur_cheque", 0.0)))
    exo = max(0.0, float(cfg_ale.get("exon_par_cheque", 6.0)))
    brut_m = nb * val
    exo_m = nb * exo
    a_compter_m = max(0.0, brut_m - exo_m)
    return r2(brut_m), r2(exo_m), r2(a_compter_m)

def revenus_annuels_apres_exonerations(revenus_annuels: list, cfg_soc: dict, cfg_ale: dict) -> float:
    total_m = 0.0
    for r in revenus_annuels:
        t = r.get("type", "standard")
        eligible = bool(r.get("eligible", True))

        if t == "ale":
            if "nb_cheques_mois" in r:
                _brut_m, _exo_m, a_compter_m = _ale_montants(r.get("nb_cheques_mois", 0), cfg_ale)
                total_m += a_compter_m
            else:
                total_m += max(0.0, float(r.get("ale_part_excedentaire_mensuel", 0.0)))
            continue

        a = max(0.0, float(r.get("montant_annuel", 0.0)))
        m = a / 12.0

        if t in ("socio_prof", "etudiant") and eligible:
            ded = min(float(cfg_soc["max_mensuel"]), m)
            total_m += max(0.0, m - ded)
        elif t == "artistique_irregulier" and eligible:
            ded_m = float(cfg_soc["artistique_annuel"]) / 12.0
            total_m += max(0.0, m - min(ded_m, m))
        else:
            total_m += m

    return float(max(0.0, total_m * 12.0))

# ============================================================
# ART.34 — MODE SIMPLE
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

def _coh_display_name(c: dict) -> str:
    return (c.get("name") or c.get("nom") or c.get("label") or "").strip()







#def cohabitants_art34_part_mensuelle_cpas(cohabitants: list,
                                         #taux_a_laisser_mensuel: float,
                                         #partage_active: bool,
                                         #nb_demandeurs_a_partager: int,
                                         #as_of: date) -> dict:
    
def cohabitants_art34_part_mensuelle_cpas(cohabitants: list,
                                         taux_a_laisser_mensuel: float,
                                         categorie: str,
                                         partage_active: bool,
                                         nb_demandeurs_a_partager: int,
                                         as_of: date) -> dict:
    taux = max(0.0, float(taux_a_laisser_mensuel))

    revenus_partenaire_m = 0.0
    nb_partenaire = 0

    revenus_debiteurs_m_brut = 0.0
    nb_debiteurs = 0
    debiteurs_excedents_m_total = 0.0

    detail_partenaire = []
    detail_debiteurs = []

    cat_norm = (categorie or "").strip().lower()

    for c in cohabitants:
        typ = normalize_art34_type(c.get("type", "autre"))
        if bool(c.get("exclure", False)):
            continue
        if not cohabitant_is_active_asof(c, as_of):
            continue

        revenu_ann = max(0.0, float(c.get("revenu_net_annuel", 0.0)))
        revenu_m = revenu_ann / 12.0
        nom = _coh_display_name(c)

        if typ == "partenaire":
            # ✅ règle partenaire
            if cat_norm == "fam_charge":
                compte_m = revenu_m
                mode = "fam_charge: 100% pris en compte"
            else:
                compte_m = max(0.0, revenu_m - taux)
                mode = "cohab/isolé: max(0, revenu - taux_cohab)"

            revenus_partenaire_m += compte_m
            nb_partenaire += 1
            detail_partenaire.append({
                "type": "partenaire",
                "name": nom,
                "mensuel_brut": r2(revenu_m),
                "taux_a_laisser_mensuel": r2(taux),
                "mensuel_pris_en_compte": r2(compte_m),
                "mensuel": r2(compte_m),  # ✅ alias pour le PDF (ton rendu lit 'mensuel')
                "regle": mode,
                "annuel": r2(revenu_ann),
            })

        elif typ in {"debiteur_direct_1", "debiteur_direct_2"}:
            revenus_debiteurs_m_brut += revenu_m
            nb_debiteurs += 1

            excedent_m = max(0.0, revenu_m - taux)
            debiteurs_excedents_m_total += excedent_m

            detail_debiteurs.append({
                "type": typ,
                "name": nom,
                "mensuel_brut": r2(revenu_m),
                "taux_a_laisser_mensuel": r2(taux),
                "mensuel_pris_en_compte": r2(excedent_m),
                "regle": "max(0, revenu - taux_cohab)",
                "annuel": r2(revenu_ann),
                "excedent_mensuel_apres_deduction": r2(excedent_m),  # compat PDF actuel
                "mensuel": r2(revenu_m),  # compat PDF actuel
            })

    debiteurs_excedents_m_total = r2(debiteurs_excedents_m_total)

    if partage_active:
        n = max(1, int(nb_demandeurs_a_partager))
        part_debiteurs_m_par_dem = r2(debiteurs_excedents_m_total / n)
    else:
        part_debiteurs_m_par_dem = r2(debiteurs_excedents_m_total)

    total_cohabitants_m = r2(revenus_partenaire_m + part_debiteurs_m_par_dem)

    return {
        "cohabitants_n_partenaire_pris_en_compte": int(nb_partenaire),
        "cohabitants_n_debiteurs_pris_en_compte": int(nb_debiteurs),
        "revenus_partenaire_mensuels_total": r2(revenus_partenaire_m),
        "revenus_debiteurs_mensuels_total": r2(revenus_debiteurs_m_brut),
        "cohabitants_part_debiteurs_avant_partage_mensuel": r2(debiteurs_excedents_m_total),
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
# MENAGE AVANCE (MULTI) — CASCADE / POOLS / PRIORITE
# ============================================================
def make_pool_key(ids: list) -> str:
    a = ",".join(sorted([str(x) for x in (ids or []) if str(x).strip()]))
    return f"ids[{a}]"

def art34_group_excess_m_cascade(debtors: list, taux: float, injected_income_m: float = 0.0) -> dict:
    """
    ✅ RÈGLE "CASCADE" (exemple CPAS) :
    On calcule une base de groupe = max(0, (somme revenus mensuels + injections) - N * taux_cat1)

    -> correspond à l'idée "chacun bénéficie fictivement d'au moins un taux cohabitant"
    -> et colle à ton exemple où on fait :
       (1.000 + 78,04) - (2 * 876,13) < 0 => 0
    """
    t = float(max(0.0, taux))
    n = max(0, len(debtors or []))
    sum_m = 0.0
    per_deb = []

    for d in (debtors or []):
        rm = max(0.0, float(d.get("revenu_net_annuel", 0.0))) / 12.0
        sum_m += rm
        per_deb.append({"name": (d.get("name") or "").strip(), "mensuel": r2(rm)})

    inj = max(0.0, float(injected_income_m))
    sum_with_inj = sum_m + inj

    base = max(0.0, sum_with_inj - n * t)

    return {
        "n_debiteurs": int(n),
        "taux": r2(t),
        "revenus_m_somme": r2(sum_m),
        "injection_m": r2(inj),
        "revenus_m_total": r2(sum_with_inj),
        "deduction_n_fois_taux": r2(n * t),
        "base_exces_m": r2(base),
        "detail_debiteurs": per_deb,
    }

#ici 

def art34_draw_from_pool_cascade(degree: int,
                                 debtor_ids: list,
                                 household: dict,
                                 taux: float,
                                 pools: dict,
                                 share_plan: dict,
                                 injected_income_m: float = 0.0,
                                 cap_take_m: float | None = None) -> dict:

    ids = list(debtor_ids or [])
    members_by_id = household.get("members_by_id", {}) or {}
    debtors = [members_by_id[i] for i in ids if i in members_by_id]

    key = make_pool_key(ids)

    # ✅ Base recalculée à chaque appel (avec injection éventuelle)
    base_info = art34_group_excess_m_cascade(debtors, taux, injected_income_m=injected_income_m)
    base = float(base_info["base_exces_m"])

    # ✅ pools[key] = "déjà pris" (taken so far), pas "reste"
    if key not in pools:
        pools[key] = 0.0

    deja_pris = float(pools[key])
    disponible = max(0.0, base - deja_pris)

    # init partage (si demandé) : plafond figé au 1er passage
    if key in share_plan and share_plan[key].get("count", 1) > 1:
        if float(share_plan[key].get("per", 0.0)) <= 0.0:
            share_plan[key]["per"] = r2(base / float(share_plan[key]["count"]))
        per = float(share_plan[key]["per"])
        take = min(disponible, per)
    else:
        take = disponible

    # ✅ cap optionnel : ne pas prendre plus que le besoin restant
    if cap_take_m is not None:
        take = min(float(take), max(0.0, float(cap_take_m)))

    take = r2(max(0.0, take))

    # ✅ on augmente "déjà pris"
    pools[key] = r2(deja_pris + take)

    reste = r2(max(0.0, base - float(pools[key])))

    return {
        "key": key,
        "degree": int(degree),
        "debtor_ids": ids,
        "base_info": base_info,
        "pool_initial_base_m": r2(base),
        "pris_en_compte_m": float(take),
        "reste_pool_m": float(reste),
        "partage_active": bool(key in share_plan and share_plan[key].get("count", 1) > 1),
        "partage_count": int(share_plan.get(key, {}).get("count", 1)),
        "partage_per_m": float(share_plan.get(key, {}).get("per", 0.0)),
    }

#ici
    #return {
        #"key": key,
        #"degree": int(degree),
        #"debtor_ids": ids,
        #"base_info": base_info,
        #"pool_initial_base_m": r2(base),
        #"pris_en_compte_m": float(take),
        #"reste_pool_m": float(pools[key]),
        #"partage_active": bool(key in share_plan and share_plan[key].get("count", 1) > 1),
        #"partage_count": int(share_plan.get(key, {}).get("count", 1)),
        #"partage_per_m": float(share_plan.get(key, {}).get("per", 0.0)),
    #}

def compute_art34_menage_avance_cascade(dossier: dict,
                                       household: dict,
                                       taux: float,
                                       pools: dict,
                                       share_plan: dict,
                                       prior_results: list,
                                       besoin_m: float | None = None) -> dict:

    """
    Cascade :
      1) essayer débiteurs 1er degré
      2) si 0 -> essayer débiteurs 2e degré
    + injection RI : on ajoute le RI mensuel des dossiers indiqués (prior_results) AU GROUPE (avant N*taux)
    """
    include_from = dossier.get("include_ris_from_dossiers", []) or []
    inj_m = 0.0
    for idx in include_from:
        if 0 <= idx < len(prior_results) and prior_results[idx] is not None:
            inj_m += float(prior_results[idx].get("ris_theorique_mensuel", 0.0))
    inj_m = r2(inj_m)
#ici
    deg1_ids = dossier.get("art34_deg1_ids", []) or []
    deg2_ids = dossier.get("art34_deg2_ids", []) or []

    dbg1 = None
    dbg2 = None
    used_degree = 0
    part_m = 0.0

    # Besoin mensuel max qu’on peut couvrir via art.34 (si fourni)
    besoin_restant = None if besoin_m is None else r2(max(0.0, float(besoin_m)))

    # 1) 1er degré d’abord (capé au besoin)
    if deg1_ids and (besoin_restant is None or besoin_restant > 0):
        dbg1 = art34_draw_from_pool_cascade(
            degree=1,
            debtor_ids=deg1_ids,
            household=household,
            taux=taux,
            pools=pools,
            share_plan=share_plan,
            injected_income_m=inj_m,
            cap_take_m=besoin_restant
        )
        take1 = float(dbg1.get("pris_en_compte_m", 0.0))
        part_m += take1
        if take1 > 0:
            used_degree = 1
        if besoin_restant is not None:
            besoin_restant = r2(max(0.0, besoin_restant - take1))

    # 2) Ensuite 2e degré si besoin restant > 0 (et si 2e degré existe)
    if deg2_ids and (besoin_restant is None or besoin_restant > 0):
        dbg2 = art34_draw_from_pool_cascade(
            degree=2,
            debtor_ids=deg2_ids,
            household=household,
            taux=taux,
            pools=pools,
            share_plan=share_plan,
            injected_income_m=0.0,   # injection uniquement au 1er degré
            cap_take_m=besoin_restant
        )
        take2 = float(dbg2.get("pris_en_compte_m", 0.0))
        part_m += take2
        if take2 > 0 and used_degree == 0:
            used_degree = 2
        if besoin_restant is not None:
            besoin_restant = r2(max(0.0, besoin_restant - take2))

    part_m = r2(part_m)
#a là

    return {
        "art34_mode": "MENAGE_AVANCE_CASCADE",
        "taux_a_laisser_mensuel": r2(taux),
        "art34_degree_utilise": int(used_degree),
        "cohabitants_part_a_compter_mensuel": float(part_m),
        "cohabitants_part_a_compter_annuel": float(r2(part_m * 12.0)),
        "debug_deg1": dbg1,
        "debug_deg2": dbg2,
        "ris_injecte_mensuel": float(inj_m),
        "include_ris_from_dossiers": list(include_from),
    }

# ============================================================
# CALCUL GLOBAL — OFFICIEL CPAS (ANNUEL puis /12)
# ============================================================
def compute_officiel_cpas_annuel(answers: dict, engine: dict, as_of=None) -> dict:
    cfg = engine["config"]
    cat = answers.get("categorie", "isole")

    taux_ris_annuel = r2(float(cfg.get("ris_rates_annuel", {}).get(cat) or 0.0))
    taux_ris_m = r2(taux_ris_annuel / 12.0) if taux_ris_annuel > 0 else 0.0

    if as_of is None:
        as_of = answers.get("date_demande", date.today())
        if not isinstance(as_of, date):
            as_of = date.today()

    revenus_demandeur_annuels = revenus_annuels_apres_exonerations(
        answers.get("revenus_demandeur_annuels", []),
        cfg["socio_prof"],
        cfg["ale"]
    )
    revenus_conjoint_annuels = 0.0
    if bool(answers.get("couple_demandeur", False)):
        revenus_conjoint_annuels = revenus_annuels_apres_exonerations(
            answers.get("revenus_conjoint_annuels", []),
            cfg["socio_prof"],
            cfg["ale"]
        )
        revenus_demandeur_annuels += revenus_conjoint_annuels
    revenus_demandeur_annuels = r2(revenus_demandeur_annuels)
    revenus_conjoint_annuels = r2(revenus_conjoint_annuels)

    #cap_detail = capital_mobilier_calc(
        #total_capital=answers.get("capital_mobilier_total", 0.0),
        #compte_commun=answers.get("capital_compte_commun", False),
        #nb_titulaires=answers.get("capital_nb_titulaires", 1),
        #categorie=cat,
        #conjoint_compte_commun=answers.get("capital_conjoint_cotitulaire", False),
        #part_fraction_custom=answers.get("capital_fraction", 1.0),
        #cfg_cap=cfg["capital_mobilier"]
    #)
    #cap_ann = r2(cap_detail["annuel"])

    #immo_detail = immo_calc_total(
        #biens=answers.get("biens_immobiliers", []),
        #enfants=answers.get("enfants_a_charge", 0),
        #cfg_immo=cfg["immo"]
    #)
    #immo_ann = r2(immo_detail["total_annuel"])

    #ces_detail = cession_biens_calc(
        #cessions=answers.get("cessions", []),
        #cas_particulier_tranche_37200=answers.get("cession_cas_particulier_37200", False),
        #dettes_deductibles=answers.get("cession_dettes_deductibles", 0.0),
        #abatt_cat=answers.get("cession_abatt_cat", "cat1"),
        #abatt_mois_prorata=answers.get("cession_abatt_mois", 0),
        #cfg_cession=cfg["cession"],
        #cfg_cap=cfg["capital_mobilier"]
    #)
    #ces_ann = r2(ces_detail["annuel"])

    # art.34 (mode simple)
    #art34 = cohabitants_art34_part_mensuelle_cpas(
        #cohabitants=answers.get("cohabitants_art34", []),
        #taux_a_laisser_mensuel=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        #partage_active=bool(answers.get("partage_enfants_jeunes_actif", False)),
        #nb_demandeurs_a_partager=int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
        #as_of=as_of
    #)

    #pf_m = r2(max(0.0, float(answers.get("prestations_familiales_a_compter_mensuel", 0.0))))
    #pf_ann = r2(pf_m * 12.0)

    #avantage_nature_m = r2(max(0.0, float(answers.get("avantage_nature_logement_mensuel", 0.0))))
    #avantage_nature_ann = r2(avantage_nature_m * 12.0)
#ici
        # ============================================================
    # ✅ Patrimoine & avantages : MENAGE (commun) + PERSO (dossier)
    # ============================================================
    pat_common = _extract_patrimoine(answers.get("_patrimoine_common"))
    pat_perso  = _extract_patrimoine(answers.get("_patrimoine_perso"))

    # --- Capitaux mobiliers ---
    cap_common_detail = capital_mobilier_calc(
        total_capital=pat_common.get("capital_mobilier_total", 0.0),
        compte_commun=pat_common.get("capital_compte_commun", False),
        nb_titulaires=pat_common.get("capital_nb_titulaires", 1),
        categorie=cat,
        conjoint_compte_commun=pat_common.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=pat_common.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    )
    cap_common_ann = r2(cap_common_detail["annuel"])

    cap_perso_detail = capital_mobilier_calc(
        total_capital=pat_perso.get("capital_mobilier_total", 0.0),
        compte_commun=pat_perso.get("capital_compte_commun", False),
        nb_titulaires=pat_perso.get("capital_nb_titulaires", 1),
        categorie=cat,
        conjoint_compte_commun=pat_perso.get("capital_conjoint_cotitulaire", False),
        part_fraction_custom=pat_perso.get("capital_fraction", 1.0),
        cfg_cap=cfg["capital_mobilier"]
    )
    cap_perso_ann = r2(cap_perso_detail["annuel"])

    cap_ann = r2(cap_common_ann + cap_perso_ann)

    # --- Immobilier ---
    immo_common_detail = immo_calc_total(
        biens=pat_common.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )
    immo_common_ann = r2(immo_common_detail["total_annuel"])

    immo_perso_detail = immo_calc_total(
        biens=pat_perso.get("biens_immobiliers", []),
        enfants=answers.get("enfants_a_charge", 0),
        cfg_immo=cfg["immo"]
    )
    immo_perso_ann = r2(immo_perso_detail["total_annuel"])

    immo_ann = r2(immo_common_ann + immo_perso_ann)

    # --- Cession de biens ---
    ces_common_detail = cession_biens_calc(
        cessions=pat_common.get("cessions", []),
        cas_particulier_tranche_37200=pat_common.get("cession_cas_particulier_37200", False),
        dettes_deductibles=pat_common.get("cession_dettes_deductibles", 0.0),
        abatt_cat=pat_common.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=pat_common.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    )
    ces_common_ann = r2(ces_common_detail["annuel"])

    ces_perso_detail = cession_biens_calc(
        cessions=pat_perso.get("cessions", []),
        cas_particulier_tranche_37200=pat_perso.get("cession_cas_particulier_37200", False),
        dettes_deductibles=pat_perso.get("cession_dettes_deductibles", 0.0),
        abatt_cat=pat_perso.get("cession_abatt_cat", "cat1"),
        abatt_mois_prorata=pat_perso.get("cession_abatt_mois", 0),
        cfg_cession=cfg["cession"],
        cfg_cap=cfg["capital_mobilier"]
    )
    ces_perso_ann = r2(ces_perso_detail["annuel"])

    ces_ann = r2(ces_common_ann + ces_perso_ann)

    # --- Avantage en nature ---
    avantage_common_m = r2(max(0.0, float(pat_common.get("avantage_nature_logement_mensuel", 0.0))))
    avantage_perso_m  = r2(max(0.0, float(pat_perso.get("avantage_nature_logement_mensuel", 0.0))))
    avantage_nature_m = r2(avantage_common_m + avantage_perso_m)

    avantage_common_ann = r2(avantage_common_m * 12.0)
    avantage_perso_ann  = r2(avantage_perso_m * 12.0)
    avantage_nature_ann = r2(avantage_nature_m * 12.0)
#jusqu'ici 
# --- Prestations familiales (PF) ---
    pf_m = r2(max(0.0, float(
        answers.get("prestations_familiales_a_compter_mensuel",
                    answers.get("pf_a_compter_mensuel", 0.0))
    )))
    pf_ann = r2(pf_m * 12.0)

    # --- Art.34 (mode simple) ---
   # art34 = cohabitants_art34_part_mensuelle_cpas(
        #cohabitants=answers.get("cohabitants_art34", []),
        #taux_a_laisser_mensuel=float(cfg["art34"].get("taux_a_laisser_mensuel", 0.0)),
        #partage_active=bool(answers.get("partage_enfants_jeunes_actif", False)),
        #nb_demandeurs_a_partager=int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
        #as_of=as_of
    #)
    art34 = cohabitants_art34_part_mensuelle_cpas(
        cohabitants=answers.get("cohabitants_art34", []),
        taux_a_laisser_mensuel=float(cfg["art34"].get("taux_a_laisser_mensuel", 0.0)),
        categorie=cat,
        partage_active=bool(answers.get("partage_enfants_jeunes_actif", False)),
        nb_demandeurs_a_partager=int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
        as_of=as_of
    )


        # --- Prestations familiales (PF) ---
    #pf_m = r2(max(0.0, float(answers.get("prestations_familiales_a_compter_mensuel", 0.0))))
    #pf_ann = r2(pf_m * 12.0)

    # --- Art.34 (mode simple) ---
    # (En ménage avancé, tu mets answers["cohabitants_art34"] = [] donc ça fera 0 — parfait)
    #art34 = cohabitants_art34_part_mensuelle_cpas(
        #cohabitants=answers.get("cohabitants_art34", []),
        #taux_a_laisser_mensuel=float(cfg["art34"]["taux_a_laisser_mensuel"]),
        #partage_active=bool(answers.get("partage_enfants_jeunes_actif", False)),
        #nb_demandeurs_a_partager=int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
        #as_of=as_of
    #)


    total_demandeur_avant_annuel = r2(
        revenus_demandeur_annuels
        + cap_ann
        + immo_ann
        + ces_ann
        + pf_ann
        + avantage_nature_ann
    )
    total_cohabitants_annuel = r2(float(art34["cohabitants_part_a_compter_annuel"]))
    total_avant_annuel = r2(total_demandeur_avant_annuel + total_cohabitants_annuel)

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
        "demandeur_nom": str(answers.get("demandeur_nom", "") or "").strip(),
        "partage_enfants_jeunes_actif": bool(answers.get("partage_enfants_jeunes_actif", False)),
        "nb_enfants_jeunes_demandeurs": int(answers.get("nb_enfants_jeunes_demandeurs", 1)),
        "revenus_demandeur_annuels": float(revenus_demandeur_annuels),
        "revenus_conjoint_annuels": float(revenus_conjoint_annuels),
        #"capitaux_mobiliers_annuels": float(cap_ann),
        #"capitaux_mobiliers_detail": cap_detail,
        #"immo_annuels": float(immo_ann),
        #"immo_detail": immo_detail,
        #"cession_biens_annuelle": float(ces_ann),
        #"cession_detail": ces_detail,
        # ✅ Totaux + détails patrimoine (commun / perso)
        "capitaux_mobiliers_annuels": float(cap_ann),
        "capitaux_mobiliers_annuels_common": float(cap_common_ann),
        "capitaux_mobiliers_annuels_perso": float(cap_perso_ann),
        "capitaux_mobiliers_detail_common": cap_common_detail,
        "capitaux_mobiliers_detail_perso": cap_perso_detail,

        "immo_annuels": float(immo_ann),
        "immo_annuels_common": float(immo_common_ann),
        "immo_annuels_perso": float(immo_perso_ann),
        "immo_detail_common": immo_common_detail,
        "immo_detail_perso": immo_perso_detail,

        "cession_biens_annuelle": float(ces_ann),
        "cession_biens_annuelle_common": float(ces_common_ann),
        "cession_biens_annuelle_perso": float(ces_perso_ann),
        "cession_detail_common": ces_common_detail,
        "cession_detail_perso": ces_perso_detail,
        **art34,
        "prestations_familiales_a_compter_mensuel": float(pf_m),
        "prestations_familiales_a_compter_annuel": float(pf_ann),
        "avantage_nature_logement_mensuel": float(avantage_nature_m),
        "avantage_nature_logement_annuel": float(avantage_nature_ann),
        "avantage_nature_logement_mensuel_common": float(avantage_common_m),
        "avantage_nature_logement_mensuel_perso": float(avantage_perso_m),
        "avantage_nature_logement_annuel_common": float(avantage_common_ann),
        "avantage_nature_logement_annuel_perso": float(avantage_perso_ann),
        #"avantage_nature_logement_mensuel": float(avantage_nature_m),
        #"avantage_nature_logement_annuel": float(avantage_nature_ann),
        "total_ressources_demandeur_avant_immunisation_annuel": float(total_demandeur_avant_annuel),
        "total_ressources_cohabitants_annuel": float(total_cohabitants_annuel),
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
            "_detail_res": res_seg,
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
# PDF — VERSION "CPAS" (mise en forme + cascade ménage avancé)
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

def cat_label(cat: str) -> str:
    cat = (cat or "").strip().lower()
    mapping = {"cohab": "Cohabitant", "isole": "Isolé", "fam_charge": "Famille à charge"}
    return mapping.get(cat, cat)

def _safe(s) -> str:
    return (s or "").replace("\n", " ").strip()

def make_decision_pdf_cpas(
    dossier_label: str,
    answers_snapshot: dict,
    res_mois_suivants: dict,
    seg_first_month=None,
    logo_path: str = "logo.png",
    cfg_soc: dict | None = None,
    cfg_ale: dict | None = None,
    cfg_cap: dict | None = None,
    cfg_immo: dict | None = None,
    cfg_cession: dict | None = None,
) -> BytesIO | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, ListFlowable, ListItem, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
    except Exception:
        return None

    cfg_soc = cfg_soc or {"max_mensuel": 0.0, "artistique_annuel": 0.0}
    cfg_ale = cfg_ale or {"valeur_cheque": 0.0, "exon_par_cheque": 6.0}
    cfg_cap = cfg_cap or DEFAULT_ENGINE["config"]["capital_mobilier"]
    cfg_immo = cfg_immo or DEFAULT_ENGINE["config"]["immo"]
    cfg_cession = cfg_cession or DEFAULT_ENGINE["config"]["cession"]

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.7*cm, rightMargin=1.7*cm, topMargin=1.2*cm, bottomMargin=1.2*cm)

    styles = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=styles["Normal"], fontName="Helvetica", fontSize=10, leading=13)
    cell = ParagraphStyle("cell", parent=base, fontSize=9.0, leading=11)
    cell_small = ParagraphStyle("cell_small", parent=base, fontSize=8.6, leading=10.6)
    small = ParagraphStyle("small", parent=base, fontSize=9, leading=12, textColor=colors.grey)

    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=18, leading=20, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=14, spaceBefore=10, spaceAfter=4)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=6, spaceAfter=2)

    story = []

    # ---------- Header ----------
    logo_elem = None
    logo_h = 3.0 * cm
    logo_w = 4.0 * cm
    if logo_path and os.path.exists(logo_path):
        logo_elem = Image(logo_path, width=logo_w, height=logo_h)
        logo_elem.hAlign = "LEFT"

    demandeur_nom = _safe(answers_snapshot.get("demandeur_nom", "")) or _safe(res_mois_suivants.get("demandeur_nom", ""))

    header_data = [
        [logo_elem if logo_elem else Paragraph("", base), Paragraph("Calcul du Revenu d’Intégration", h1)],
        ["", Paragraph(f"Dossier : <b>{_safe(dossier_label)}</b>", base)],
    ]
    if demandeur_nom:
        header_data.append(["", Paragraph(f"Demandeur : <b>{demandeur_nom}</b>", base)])

    header_tbl = Table(header_data, colWidths=[3.1*cm, 13.2*cm])
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (0, 0), -34),
        ("TOPPADDING",  (0, 0), (0, 0), -18),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        f"Catégorie : <b>{cat_label(res_mois_suivants.get('categorie',''))}</b> — "
        f"Taux RI annuel (référence) : <b>{euro(res_mois_suivants.get('taux_ris_annuel',0))} €</b>",
        base
    ))
    story.append(Paragraph(f"Taux RI mensuel (dérivé) : <b>{euro(res_mois_suivants.get('taux_ris_mensuel_derive',0))} €</b>", base))
    story.append(Spacer(1, 10))

    def bullets(lines: list[str]):
        items = [ListItem(Paragraph(l, base), leftIndent=12) for l in lines]
        return ListFlowable(items, bulletType="bullet", start="•", leftIndent=14)

    def money_table(rows: list[list[str]], col_widths=None):
        if not rows:
            return Paragraph("", base)
        conv = []
        for ridx, r in enumerate(rows):
            row_conv = []
            for v in r:
                txt = "" if v is None else str(v)
                stl = cell if ridx == 0 else cell_small
                row_conv.append(Paragraph(txt.replace("\n", "<br/>"), stl))
            conv.append(row_conv)

        tbl = Table(conv, colWidths=col_widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return tbl

    def revenu_detail_line(r: dict) -> str:
        typ = (r.get("type") or "standard").strip()
        eligible = bool(r.get("eligible", True))
        if typ == "ale":
            nb = float(r.get("nb_cheques_mois", 0.0))
            brut_m, exo_m, a_compter_m = _ale_montants(nb, cfg_ale)
            val = float(cfg_ale.get("valeur_cheque", 0.0))
            exo = float(cfg_ale.get("exon_par_cheque", 6.0))
            return (
                f"ALE : {nb:g} chq/mois × {euro(val)} € = {euro(brut_m)} €/mois ; "
                f"exo {nb:g} × {euro(exo)} € = {euro(exo_m)} €/mois ; "
                f"à compter = {euro(a_compter_m)} €/mois (soit {euro(a_compter_m*12)} €/an)"
            )

        a = float(r.get("montant_annuel", 0.0))
        m = a / 12.0

        if typ in ("socio_prof", "etudiant"):
            if not eligible:
                return f"{typ} : non éligible → montant compté = {euro(a)} €/an"
            ded = min(float(cfg_soc.get("max_mensuel", 0.0)), m)
            compt_m = max(0.0, m - ded)
            return f"{typ} : {euro(m)} €/mois − déduction {euro(ded)} €/mois = {euro(compt_m)} €/mois (×12)"

        if typ == "artistique_irregulier":
            if not eligible:
                return f"artistique irrégulier : non éligible → montant compté = {euro(a)} €/an"
            ded_m_ref = float(cfg_soc.get("artistique_annuel", 0.0)) / 12.0
            ded = min(ded_m_ref, m)
            compt_m = max(0.0, m - ded)
            return f"artistique irrégulier : {euro(m)} €/mois − déduction {euro(ded)} €/mois = {euro(compt_m)} €/mois (×12)"

        return f"standard : {euro(a)} €/an (soit {euro(m)} €/mois)"

    def render_revenus_block(title: str, revenus_list: list) -> float:
        story.append(Paragraph(title, h3))
        if not revenus_list:
            story.append(Paragraph("Aucun revenu encodé.", base))
            return 0.0

        rows = [["Type/label", "Règle", "Calcul (détail)", "Montant compté (annuel)"]]
        total_ann = 0.0

        for r in revenus_list:
            label = _safe(r.get("label", ""))
            typ = _safe(r.get("type", "standard"))

            if typ == "ale" and "nb_cheques_mois" in r:
                nb = float(r.get("nb_cheques_mois", 0.0))
                _brut_m, _exo_m, a_compter_m = _ale_montants(nb, cfg_ale)
                compt_ann = float(a_compter_m) * 12.0
            else:
                a = max(0.0, float(r.get("montant_annuel", 0.0)))
                m = a / 12.0
                eligible = bool(r.get("eligible", True))
                if typ in ("socio_prof", "etudiant") and eligible:
                    ded = min(float(cfg_soc.get("max_mensuel", 0.0)), m)
                    compt_ann = max(0.0, (m - ded) * 12.0)
                elif typ == "artistique_irregulier" and eligible:
                    ded_m = float(cfg_soc.get("artistique_annuel", 0.0)) / 12.0
                    compt_ann = max(0.0, (m - min(ded_m, m)) * 12.0)
                elif typ == "ale":
                    compt_ann = max(0.0, float(r.get("ale_part_excedentaire_mensuel", 0.0))) * 12.0
                else:
                    compt_ann = a

            total_ann += float(compt_ann)
            rows.append([label, typ, revenu_detail_line(r), f"{euro(compt_ann)} €"])

        story.append(money_table(rows, col_widths=[5.0*cm, 2.2*cm, 6.3*cm, 3.7*cm]))
        story.append(Spacer(1, 4))

        story.append(Paragraph(
            f"<font size=9 color='grey'>"
            f"Exo socio-pro max : {euro(cfg_soc.get('max_mensuel',0))} €/mois — "
            f"Exo artistique irrégulier : {euro(cfg_soc.get('artistique_annuel',0))} €/an — "
            f"ALE : valeur chèque = {euro(cfg_ale.get('valeur_cheque',0))} € ; exonération = {euro(cfg_ale.get('exon_par_cheque',6))} €/chèque"
            f"</font>",
            small
        ))
        story.append(Paragraph(f"<font size=9 color='grey'>Total revenus comptés (annuel) pour ce bloc : {euro(total_ann)} €</font>", small))
        return float(r2(total_ann))

    def render_cohabitants_block(cohabitants: list, res_seg: dict):
        story.append(Paragraph("Ressources des cohabitants (art.34) :", h3))

        # --- CASCADE (ménage avancé) ---
        if str(res_seg.get("art34_mode", "")).upper() == "MENAGE_AVANCE_CASCADE":
            taux = float(res_seg.get("taux_a_laisser_mensuel", 0.0))  # peut être 0 si non stocké, on reprend du debug
            dbg1 = res_seg.get("debug_deg1")
            dbg2 = res_seg.get("debug_deg2")

            lines = []
            lines.append("<b>Règle de priorité (cascade) :</b> partenaire (déjà pris en compte), puis débiteurs 1er degré, puis 2e degré si 1er degré = 0.")
            if float(res_seg.get("ris_injecte_mensuel", 0.0)) > 0:
                lines.append(f"Injection RI mensuel (autres dossiers) : <b>{euro(res_seg.get('ris_injecte_mensuel',0))} €</b> (ajoutée au groupe 1er degré avant N×taux).")

            def _block_dbg(dbg: dict, label_deg: str):
                bi = (dbg.get("base_info") or {})
                n = int(bi.get("n_debiteurs", 0))
                taux_loc = float(bi.get("taux", 0.0))
                t = euro(taux_loc)
                lines.append(f"<b>{label_deg}</b>")
                # détail débiteurs
                for d in (bi.get("detail_debiteurs") or []):
                    nm = _safe(d.get("name","")) or "Débiteur"
                    lines.append(f"— {nm} : {euro(d.get('mensuel',0))} €/mois")
                # formule groupe
                lines.append(
                    f"{euro(bi.get('revenus_m_somme',0))} €"
                    + (f" + {euro(bi.get('injection_m',0))} €" if float(bi.get('injection_m',0)) > 0 else "")
                    + f" − ({n} × {t} €) = {euro(bi.get('base_exces_m',0))} €"
                )
                if dbg.get("partage_active"):
                    lines.append(
                        f"Partage pool : {euro(dbg.get('pool_initial_base_m',0))} € / {int(dbg.get('partage_count',1))}"
                        f" = {euro(dbg.get('partage_per_m',0))} € (plafond par dossier)"
                    )
                lines.append(f"Pris en compte : <b>{euro(dbg.get('pris_en_compte_m',0))} €</b> / mois")
                lines.append(f"Reste pool : {euro(dbg.get('reste_pool_m',0))} € / mois")

            if dbg1:
                _block_dbg(dbg1, "Débiteurs 1er degré :")
            if dbg2:
                _block_dbg(dbg2, "Débiteurs 2e degré :")

            lines.append(
                f"Total cohabitants compté : {euro(float(res_seg.get('cohabitants_part_a_compter_mensuel',0)))} € × 12"
                f" = {euro(float(res_seg.get('cohabitants_part_a_compter_annuel',0)))} €"
            )

            story.append(bullets(lines))
            return

        # --- MODE SIMPLE (inchangé) ---
        active_info = []
        for c in cohabitants or []:
            rev_ann = float(c.get("revenu_net_annuel", 0.0))
            dq = c.get("date_quitte_menage")
            excl = bool(c.get("exclure", False))
            dq_txt = f" (départ: {date_fr(dq)})" if dq else ""
            who = (c.get("name") or "").strip() or normalize_art34_type(c.get("type","autre"))
            if excl:
                active_info.append(f"{who} — {euro(rev_ann)} €/an — EXCLU{dq_txt}")
            else:
                active_info.append(f"{who} — {euro(rev_ann)} €/an{dq_txt}")

        if active_info:
            story.append(bullets(active_info))

        taux = float(res_seg.get("taux_a_laisser_mensuel", 0.0))
        part_m = float(res_seg.get("cohabitants_part_a_compter_mensuel", 0.0))
        part_ann = float(res_seg.get("cohabitants_part_a_compter_annuel", 0.0))
        detail_deb = res_seg.get("detail_debiteurs", []) or []
        detail_part = res_seg.get("detail_partenaire", []) or []

        lines = []
        if part_m <= 0:
            lines.append("Pas de ressource cohabitant prise en compte pour la période.")
        else:
            #if detail_part:
                #for p in detail_part:
                    #who = (p.get("name") or "").strip()
                    #who = f"{who} (partenaire)" if who else "partenaire"
                    #brut = float(p.get("mensuel_brut", 0.0))
                    #pris = float(p.get("mensuel_pris_en_compte", p.get("mensuel", 0.0)))
                    #regle = _safe(p.get("regle", ""))
                    #lines.append(f"{who} : brut {euro(brut)} €/mois → pris en compte {euro(pris)} €/mois ({regle})")
            if detail_part:
                for p in detail_part:
                    who = (p.get("name") or "").strip()
                    who = f"{who} (partenaire)" if who else "partenaire"

                    brut = float(p.get("mensuel_brut", p.get("mensuel", 0.0)))
                    pris = float(p.get("mensuel_pris_en_compte", 0.0))
                    regle = (p.get("regle") or "").strip()

                    lines.append(f"{who} : {euro(brut)} €/mois → pris en compte {euro(pris)} €/mois ({regle})")

            if detail_deb:
                lines.append(f"Débiteurs (déduction {euro(taux)} €/mois appliquée individuellement) :")
                for d in detail_deb:
                    who = (d.get("name") or "").strip()
                    typ = d.get("type", "")
                    who = f"{who} ({typ})" if who else typ
                    rm = float(d.get("mensuel", 0.0))
                    ex = float(d.get("excedent_mensuel_apres_deduction", 0.0))
                    lines.append(f"— {who} : {euro(rm)} − {euro(taux)} = {euro(ex)} €/mois")

            lines.append(f"Total cohabitants compté : {euro(part_m)} € × 12 = {euro(part_ann)} €")

        story.append(bullets(lines))

    # --- Rendu détail capitaux / immo / cession (depuis un dict "res-like") ---
    #def render_capitaux_detail_from(det: dict, annuel: float, title: str):
        #det = det or {}
        #details = det.get("tranches", []) or []
        #if float(annuel) <= 0 and len(details) == 0 and float(det.get("total_capital", 0.0)) <= 0:
            #return
    def render_capitaux_detail_from(det: dict, annuel: float, title: str):
        det = det or {}
        tr = det.get("tranches", []) or []

        total_cap = float(det.get("total_capital", 0.0) or 0.0)
        annuel = float(annuel or 0.0)

        # ✅ si tout est vide/0 => on n'affiche rien
        if total_cap <= 0 and annuel <= 0:
            # tranches parfois présentes même si tout = 0 -> on vérifie que tout est à zéro
            if all(float(t.get("base", 0.0) or 0.0) <= 0 and float(t.get("produit", 0.0) or 0.0) <= 0 for t in tr):
                return


    #def render_capitaux_detail_from(det: dict, annuel: float, title: str):
        #details = (res_seg or {}).get("details_capitaux") or []
        #if montant_annuel <= 0 and len(details) == 0:
            #return
        #if annuel <= 0 and float(det.get("total_capital", 0.0)) <= 0:
            #story.append(Paragraph(f"{title} : aucun.", base))
            #return

        story.append(Paragraph(f"{title} — détail :", h3))
        story.append(bullets([
            f"Total capitaux encodés : {euro(det.get('total_capital',0))} €",
            f"Fraction appliquée : {euro(det.get('fraction',0))} ({_safe(det.get('fraction_mode',''))})",
            f"Capital pris en compte (base) : {euro(det.get('capital_pris_en_compte_base',0))} €",
        ]))
        tr = det.get("tranches", []) or []
        rows2 = [["Tranche", "Base", "Taux", "Calcul", "Produit"]]
        for t in tr:
            rows2.append([
                _safe(t.get("borne", t.get("label", ""))),
                f"{euro(t.get('base',0))} €",
                f"{float(t.get('taux',0))*100:.2f} %",
                f"{euro(t.get('base',0))} × {float(t.get('taux',0))*100:.2f} %",
                f"{euro(t.get('produit',0))} €",
            ])
        story.append(money_table(rows2, col_widths=[4.0*cm, 3.0*cm, 2.2*cm, 4.7*cm, 2.9*cm]))
        story.append(Paragraph(f"<font size=9 color='grey'>Total à compter (annuel) : {euro(annuel)} €</font>", small))

    def render_immo_detail_from(det: dict, total: float, title: str):
        det = det or {}
        details = det.get("details", []) or []
        if float(total) <= 0 and len(details) == 0:
            return

    #def render_immo_detail_from(det: dict, total: float, title: str):
        #details = (res_seg or {}).get("details_immo") or []
        #if montant_annuel <= 0 and len(details) == 0:
            #return
        #if total <= 0 and not (det.get("details") or []):
            #story.append(Paragraph(f"{title} : aucun.", base))
            #return
        story.append(Paragraph(f"{title} — détail :", h3))
        story.append(bullets([
            f"Coefficient RC : {euro(det.get('coeff_rc',0))}",
            f"Exonération bâti totale : {euro(det.get('exo_bati_total',0))} € (répartie sur {int(det.get('nb_bati',0))} bien(s) bâti(s))",
            f"Exonération non bâti totale : {euro(det.get('exo_non_bati_total',0))} € (répartie sur {int(det.get('nb_non_bati',0))} bien(s) non bâti(s))",
        ]))
        rows = [["Bien", "Type", "RC", "Frac.", "RC part", "Exo", "RC-Exo", "×coeff", "Déduc.", "Pris en compte"]]
        for d in (det.get("details") or []):
            ded = float(d.get("ded_interets", 0.0)) + float(d.get("ded_rente", 0.0))
            rows.append([
                str(d.get("bien", "")),
                _safe(d.get("type", "")),
                f"{euro(d.get('rc_non_indexe',0))}",
                f"{float(d.get('fraction',0)):.2f}",
                f"{euro(d.get('rc_part',0))}",
                f"{euro(d.get('exo_par_bien',0))}",
                f"{euro(d.get('rc_apres_exo',0))}",
                f"{euro(d.get('base_coeff',0))}",
                f"{euro(ded)}",
                f"{euro(d.get('pris_en_compte',0))}",
            ])
        story.append(money_table(rows, col_widths=[1.1*cm, 1.8*cm, 1.7*cm, 1.1*cm, 1.7*cm, 1.4*cm, 1.6*cm, 1.4*cm, 1.3*cm, 2.0*cm]))
        story.append(Paragraph(f"<font size=9 color='grey'>Total à compter (annuel) : {euro(total)} €</font>", small))

    #def render_cession_detail_from(det: dict, total: float, title: str):
        #if total <= 0 and not (det.get("details_cessions") or []):
            #story.append(Paragraph(f"{title} : aucune.", base))
            #return
    def render_cession_detail_from(det: dict, total: float, title: str):
        details = (det or {}).get("details_cessions") or []
        if total <= 0 and len(details) == 0:
            return  # ✅ n'affiche rien du tout

        # ... le reste de ta fonction (tableau, puces, tranches, etc.)


        story.append(Paragraph(f"{title} — détail :", h3))
        rows = [["Cession", "Valeur vénale", "Usufruit ?", "Ratio", "Valeur retenue"]]
        for c in (det.get("details_cessions") or []):
            rows.append([
                str(c.get("cession", "")),
                f"{euro(c.get('valeur_venale',0))} €",
                "Oui" if c.get("usufruit") else "Non",
                f"{float(c.get('ratio_usufruit',0)):.2f}",
                f"{euro(c.get('valeur_retendue',0))} €",
            ])
        story.append(money_table(rows, col_widths=[1.5*cm, 3.2*cm, 2.1*cm, 2.0*cm, 6.8*cm]))
        story.append(Spacer(1, 4))

        lines = [
            f"Brut retenu : {euro(det.get('brut_total',0))} €",
            f"Dettes déductibles : {euro(det.get('dettes_deductibles',0))} € → après dettes : {euro(det.get('apres_dettes',0))} €",
        ]
        if det.get("cas_tranche_37200"):
            lines.append(f"Tranche immunisée (cas particulier) : {euro(det.get('tranche_37200',0))} € → après tranche : {euro(det.get('apres_tranche_37200',0))} €")
        lines.append(f"Abattement {det.get('abatt_cat','')} : {euro(det.get('abatt_annuel',0))} €/an, prorata {int(det.get('abatt_mois',0))}/12 = {euro(det.get('abatt_prorata',0))} €")
        lines.append(f"Base capitaux : {euro(det.get('base_cap',0))} €")
        story.append(bullets(lines))

        tr = det.get("tranches", []) or []
        rows2 = [["Tranche", "Base", "Taux", "Calcul", "Produit"]]
        for t in tr:
            rows2.append([
                _safe(t.get("borne", t.get("label", ""))),
                f"{euro(t.get('base',0))} €",
                f"{float(t.get('taux',0))*100:.2f} %",
                f"{euro(t.get('base',0))} × {float(t.get('taux',0))*100:.2f} %",
                f"{euro(t.get('produit',0))} €",
            ])
        story.append(money_table(rows2, col_widths=[4.0*cm, 3.0*cm, 2.2*cm, 4.7*cm, 2.9*cm]))
        story.append(Paragraph(f"<font size=9 color='grey'>Total à compter (annuel) : {euro(total)} €</font>", small))

    def render_totaux_block(res_seg: dict):
        total_dem = float(res_seg.get("total_ressources_demandeur_avant_immunisation_annuel", 0.0))
        total_coh = float(res_seg.get("total_ressources_cohabitants_annuel", 0.0))
        total_av = float(res_seg.get("total_ressources_avant_immunisation_simple_annuel", 0.0))
        immu = float(res_seg.get("immunisation_simple_annuelle", 0.0))
        total_ap = float(res_seg.get("total_ressources_apres_immunisation_simple_annuel", 0.0))

        rows = [
            ["Synthèse ressources", ""],
            ["Total ressources demandeur (propres)", f"{euro(total_dem)} €"],
            ["Total ressources cohabitants (art.34)", f"{euro(total_coh)} €"],
            ["Total ressources avant immunisation (addition)", f"{euro(total_av)} €"],
            ["Immunisation simple", f"{euro(immu)} €"],
            ["Total ressources après immunisation", f"{euro(total_ap)} €"],
        ]
        tbl = Table(rows, colWidths=[10.0*cm, 6.2*cm])
        from reportlab.lib import colors as _c
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND", (0,0), (-1,0), _c.lightgrey),
            ("LINEBELOW", (0,0), (-1,0), 0.6, _c.black),
            ("BOX", (0,0), (-1,-1), 0.6, _c.black),
            ("INNERGRID", (0,0), (-1,-1), 0.25, _c.grey),
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

        story.append(Paragraph("Ressources à considérer :", h2))
        story.append(Paragraph("Ressources du demandeur (propres) :", h3))

        _ = render_revenus_block("Revenus demandeur", answers_snapshot.get("revenus_demandeur_annuels", []))
        if bool(answers_snapshot.get("couple_demandeur", False)):
            _ = render_revenus_block("Revenus conjoint (si demande couple)", answers_snapshot.get("revenus_conjoint_annuels", []))

        #story.append(Spacer(1, 4))
        #render_capitaux_detail_from(res_seg.get("capitaux_mobiliers_detail") or {}, float(res_seg.get("capitaux_mobiliers_annuels", 0.0)), "Capitaux mobiliers (ménage)")
        #story.append(Spacer(1, 4))
        #render_immo_detail_from(res_seg.get("immo_detail") or {}, float(res_seg.get("immo_annuels", 0.0)), "Immobilier (RC) (ménage)")
        #story.append(Spacer(1, 4))
        #render_cession_detail_from(res_seg.get("cession_detail") or {}, float(res_seg.get("cession_biens_annuelle", 0.0)), "Cession de biens (ménage)")

        # ✅ Patrimoine : commun vs perso (affichage conditionnel)
        cap_c = float(res_seg.get("capitaux_mobiliers_annuels_common", 0.0))
        cap_p = float(res_seg.get("capitaux_mobiliers_annuels_perso", 0.0))
        render_capitaux_detail_from(res_seg.get("capitaux_mobiliers_detail_common") or {}, cap_c, "Capitaux mobiliers (ménage — commun)")
        if cap_p > 0 or float((res_seg.get("capitaux_mobiliers_detail_perso") or {}).get("total_capital", 0.0)) > 0:
            story.append(Spacer(1, 4))
            render_capitaux_detail_from(res_seg.get("capitaux_mobiliers_detail_perso") or {}, cap_p, "Capitaux mobiliers (personnels — dossier)")

        story.append(Spacer(1, 4))
        im_c = float(res_seg.get("immo_annuels_common", 0.0))
        im_p = float(res_seg.get("immo_annuels_perso", 0.0))
        render_immo_detail_from(res_seg.get("immo_detail_common") or {}, im_c, "Immobilier (RC) (ménage — commun)")
        if im_p > 0 or ((res_seg.get("immo_detail_perso") or {}).get("details") or []):
            story.append(Spacer(1, 4))
            render_immo_detail_from(res_seg.get("immo_detail_perso") or {}, im_p, "Immobilier (RC) (personnels — dossier)")

        story.append(Spacer(1, 4))
        ce_c = float(res_seg.get("cession_biens_annuelle_common", 0.0))
        ce_p = float(res_seg.get("cession_biens_annuelle_perso", 0.0))
        render_cession_detail_from(res_seg.get("cession_detail_common") or {}, ce_c, "Cession de biens (ménage — commun)")
        if ce_p > 0 or ((res_seg.get("cession_detail_perso") or {}).get("details_cessions") or []):
            story.append(Spacer(1, 4))
            render_cession_detail_from(res_seg.get("cession_detail_perso") or {}, ce_p, "Cession de biens (personnels — dossier)")

        
        pf_ann = float(res_seg.get("prestations_familiales_a_compter_annuel", 0.0))
        #avn_ann = float(res_seg.get("avantage_nature_logement_annuel", 0.0))
        avn_c_ann = float(res_seg.get("avantage_nature_logement_annuel_common", 0.0))
        avn_p_ann = float(res_seg.get("avantage_nature_logement_annuel_perso", 0.0))
        avn_ann = float(res_seg.get("avantage_nature_logement_annuel", 0.0))
        
        story.append(Spacer(1, 4))
        #story.append(bullets([
            #f"Prestations familiales : {euro(pf_ann)} € (annuel) [= {euro(float(res_seg.get('prestations_familiales_a_compter_mensuel',0)))} €/mois × 12]",
            #f"Avantage en nature logement (ménage) : {euro(avn_ann)} € (annuel) [= {euro(float(res_seg.get('avantage_nature_logement_mensuel',0)))} €/mois × 12]",
        
            #f"Avantage en nature logement : {euro(avn_ann)} € (annuel) "
            #f"[commun {euro(avn_c_ann)} € + perso {euro(avn_p_ann)} €]",

        #]))
        EPS = 0.01  # ou 0.005 selon ton r2
        
        bul = []

        #if pf_ann > 0:
        if pf_ann > EPS:

            bul.append(f"Prestations familiales : {euro(pf_ann)} € (annuel)")

        if avn_ann > 0:
            bul.append(
                f"Avantage en nature logement : {euro(avn_ann)} € (annuel) "
                f"[commun {euro(avn_c_ann)} € + perso {euro(avn_p_ann)} €]"
            )

        if bul:
            story.append(bullets(bul))


        story.append(Spacer(1, 8))
        render_cohabitants_block(answers_snapshot.get("cohabitants_art34", []), res_seg)
        story.append(Spacer(1, 8))

        render_totaux_block(res_seg)
        story.append(Spacer(1, 10))
        render_ri_block(res_seg, seg_info, seg_all)
        story.append(Spacer(1, 8))

    if seg_first_month and seg_first_month.get("segments"):
        for idx, s in enumerate(seg_first_month["segments"]):
            res_seg = s.get("_detail_res") if isinstance(s.get("_detail_res"), dict) else res_mois_suivants
            title_period = f"Du {date_fr(s['du'])} au {date_fr(s['au'])} :"
            render_one_period(title_period, res_seg, s, seg_first_month)
            if idx < len(seg_first_month["segments"]) - 1:
                story.append(PageBreak())

        story.append(Paragraph(f"--&gt; Soit un montant total de <b>{euro(seg_first_month.get('ris_1er_mois_total',0))} €</b> pour le mois concerné", base))
        story.append(Spacer(1, 6))

        ris_ms = float(seg_first_month.get("ris_mois_suivants", 0.0))
        ref_ms = seg_first_month.get("reference_mois_suivants", "")
        story.append(Paragraph(
            f"<b>Montant total à partir du mois suivant :</b> {euro(ris_ms)} € / mois "
            f"(<font size=9 color='grey'>= {euro(ris_ms*12)} € / an</font>)"
            + (f" <font size=9 color='grey'>(référence : {date_fr(ref_ms)})</font>" if ref_ms else ""),
            base
        ))
    else:
        render_one_period("Mois complet :", res_mois_suivants, None, None)

    story.append(Spacer(1, 10))
    story.append(Paragraph("Document généré automatiquement — à valider selon la décision du CPAS.", small))

    doc.build(story)
    buf.seek(0)
    return buf

# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(page_title="Calcul RIS", layout="centered")

if os.path.exists("logo.png"):
    st.image("logo.png", use_container_width=False)

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
    st.write("**ALE (chèques)** ✅")
    cfg["ale"]["valeur_cheque"] = st.number_input(
        "Valeur d'un chèque ALE (€)",
        min_value=0.0,
        value=float(cfg["ale"].get("valeur_cheque", 0.0)),
        format="%.2f"
    )
    cfg["ale"]["exon_par_cheque"] = st.number_input(
        "Exonération par chèque ALE (€)",
        min_value=0.0,
        value=float(cfg["ale"].get("exon_par_cheque", 6.0)),
        format="%.2f"
    )
    st.caption("ALE compté (mensuel) = nb chèques/mois × (valeur chèque − exon/chèque), borné à 0.")

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
    nb = st.number_input("Nombre de revenus à encoder", min_value=0, value=1, step=1, key=f"{prefix}_nb")
    for i in range(int(nb)):
        st.markdown(f"**Revenu {i+1}**")
        c1, c2, c3 = st.columns([2, 1, 1])

        label = c1.text_input("Type/label", value="salaire/chômage", key=f"{prefix}_lab_{i}")
        typ = c3.selectbox(
            "Règle",
            ["standard", "socio_prof", "etudiant", "artistique_irregulier", "ale", "prestations_familiales"],
            key=f"{prefix}_t_{i}"
        )

        if typ == "ale":
            nb_chq = c2.number_input("Nb chèques / mois", min_value=0, value=0, step=1, key=f"{prefix}_ale_n_{i}")
            brut_m, exo_m, a_compter_m = _ale_montants(nb_chq, cfg["ale"])
            st.caption(
                f"ALE (calcul) : {nb_chq} × {cfg['ale']['valeur_cheque']:.2f} € = {brut_m:.2f} €/mois ; "
                f"exo {nb_chq} × {cfg['ale']['exon_par_cheque']:.2f} € = {exo_m:.2f} €/mois ; "
                f"à compter = {a_compter_m:.2f} €/mois (= {a_compter_m*12:.2f} €/an)"
            )
            lst.append({
                "label": label,
                "type": "ale",
                "nb_cheques_mois": int(nb_chq),
                "ale_part_excedentaire_mensuel": float(a_compter_m),
                "eligible": True
            })
            continue

        montant_annuel, _p = ui_money_period_input("Montant net", key_prefix=f"{prefix}_money_{i}", default=0.0, step=100.0)
        eligible = True
        if typ in ("socio_prof", "etudiant", "artistique_irregulier"):
            eligible = st.checkbox("Éligible exonération ?", value=True, key=f"{prefix}_el_{i}")

        lst.append({
            "label": label,
            "montant_annuel": float(montant_annuel),
            "type": typ,
            "eligible": eligible,
        })
    return lst

# ============================================================
# ✅ “patrimoine like mode simple” (4 blocs)
# ============================================================

def ui_patrimoine_like_simple(prefix: str) -> dict:
    out = {}

    st.markdown("### Capitaux mobiliers")
    a_cap = st.checkbox("Le ménage possède des capitaux mobiliers", value=False, key=f"{prefix}_cap_yes")
    out["capital_mobilier_total"] = 0.0
    out["capital_compte_commun"] = False
    out["capital_nb_titulaires"] = 1
    out["capital_conjoint_cotitulaire"] = False
    out["capital_fraction"] = 1.0

    if a_cap:
        out["capital_mobilier_total"] = st.number_input(
            "Montant total capitaux (€)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_cap_total"
        )
        out["capital_compte_commun"] = st.checkbox("Compte commun ?", value=False, key=f"{prefix}_cap_cc")
        if out["capital_compte_commun"]:
            out["capital_nb_titulaires"] = st.number_input("Nombre de titulaires", min_value=1, value=2, step=1, key=f"{prefix}_cap_nbtit")
        else:
            out["capital_fraction"] = st.number_input("Part (0–1)", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key=f"{prefix}_cap_frac")

    st.divider()
    st.markdown("### Biens immobiliers")
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
    out["biens_immobiliers"] = biens

    st.divider()
    st.markdown("### Cession de biens")
    cessions = []
    a_ces = st.checkbox("Le ménage a cédé des biens (10 dernières années)", value=False, key=f"{prefix}_ces_yes")
    out["cessions"] = []
    out["cession_cas_particulier_37200"] = False
    out["cession_dettes_deductibles"] = 0.0
    out["cession_abatt_cat"] = "cat1"
    out["cession_abatt_mois"] = 0

    if a_ces:
        out["cession_cas_particulier_37200"] = st.checkbox("Cas particulier: tranche immunisée 37.200€", value=False, key=f"{prefix}_ces_37200")
        dettes_ok = st.checkbox("Déduire des dettes personnelles ?", value=False, key=f"{prefix}_ces_det_ok")
        if dettes_ok:
            out["cession_dettes_deductibles"] = st.number_input("Dettes déductibles (€)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_ces_det")
        out["cession_abatt_cat"] = st.selectbox("Catégorie d’abattement", ["cat1", "cat2", "cat3"], key=f"{prefix}_ces_cat")
        out["cession_abatt_mois"] = st.number_input("Prorata mois", min_value=0, max_value=12, value=0, step=1, key=f"{prefix}_ces_mois")

        nb_c = st.number_input("Nombre de cessions", min_value=0, value=1, step=1, key=f"{prefix}_ces_n")
        for i in range(int(nb_c)):
            st.markdown(f"**Cession {i+1}**")
            val = st.number_input("Valeur vénale (€)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_ces_v_{i}")
            usuf = st.checkbox("Usufruit ?", value=False, key=f"{prefix}_ces_u_{i}")
            cessions.append({"valeur_venale": float(val), "usufruit": bool(usuf)})

        out["cessions"] = cessions

    st.divider()
    st.markdown("### Avantage en nature")
    out["avantage_nature_logement_mensuel"] = st.number_input(
        "Logement payé par un tiers non cohabitant (€/mois) — montant à compter",
        min_value=0.0, value=0.0, step=10.0,
        key=f"{prefix}_avn"
    )

    return out

# ============================================================
# ✅ Patrimoine : ménage (commun) + perso (par dossier)
#    -> on calcule chaque bloc séparément (seuils propres) puis on additionne.
# ============================================================
PATRIMOINES_KEYS = {
    # capitaux
    "capital_mobilier_total", "capital_compte_commun", "capital_nb_titulaires",
    "capital_conjoint_cotitulaire", "capital_fraction",
    # immo
    "biens_immobiliers",
    # cession
    "cessions", "cession_cas_particulier_37200", "cession_dettes_deductibles",
    "cession_abatt_cat", "cession_abatt_mois",
    # avantage nature
    "avantage_nature_logement_mensuel",
}

def _pat_default() -> dict:
    return {
        "capital_mobilier_total": 0.0,
        "capital_compte_commun": False,
        "capital_nb_titulaires": 1,
        "capital_conjoint_cotitulaire": False,
        "capital_fraction": 1.0,
        "biens_immobiliers": [],
        "cessions": [],
        "cession_cas_particulier_37200": False,
        "cession_dettes_deductibles": 0.0,
        "cession_abatt_cat": "cat1",
        "cession_abatt_mois": 0,
        "avantage_nature_logement_mensuel": 0.0,
    }

def _extract_patrimoine(d: dict | None) -> dict:
    base = _pat_default()
    d = d or {}
    for k in PATRIMOINES_KEYS:
        if k in d:
            base[k] = d[k]
    # sécurisations minimales
    base["biens_immobiliers"] = list(base.get("biens_immobiliers") or [])
    base["cessions"] = list(base.get("cessions") or [])
    base["capital_mobilier_total"] = float(base.get("capital_mobilier_total") or 0.0)
    base["avantage_nature_logement_mensuel"] = float(base.get("avantage_nature_logement_mensuel") or 0.0)
    return base


# ============================================================
# UI Ménage commun (mode simple art34 + 4 blocs)
# ============================================================
# de la 
    
def ui_menage_common(prefix: str, nb_demandeurs: int, enable_pf_links: bool, show_simple_art34: bool = True) -> dict:
    answers = {}
    st.divider()

    answers["partage_enfants_jeunes_actif"] = False
    answers["nb_enfants_jeunes_demandeurs"] = 1

    if show_simple_art34 and nb_demandeurs > 1:
        answers["partage_enfants_jeunes_actif"] = st.checkbox(
            "Partager la part entre plusieurs ENFANTS/JEUNES demandeurs (uniquement dans ce cas)",
            value=False,
            key=f"{prefix}_partage"
        )
        if answers["partage_enfants_jeunes_actif"]:
            answers["nb_enfants_jeunes_demandeurs"] = st.number_input(
                "Nombre de demandeurs à partager",
                min_value=1, value=max(2, nb_demandeurs), step=1,
                key=f"{prefix}_nb_partage"
            )

    cohabitants = []
    pf_links = []

    if show_simple_art34:
        st.markdown("### Cohabitants admissibles (art.34) — mode simple")
        st.caption("Tu peux encoder la date de départ du ménage. Après cette date, la personne ne compte plus.")
        nb_coh = st.number_input("Nombre de cohabitants à encoder", min_value=0, value=2, step=1, key=f"{prefix}_nbcoh")

        for i in range(int(nb_coh)):
            st.markdown(f"**Cohabitant {i+1}**")
            c1, c2, c3 = st.columns([2, 1, 1])

            nom = c1.text_input("Nom (optionnel)", value="", key=f"{prefix}_coh_name_{i}")
            typ = c1.selectbox(
                "Type",
                ["partenaire", "debiteur_direct_1", "debiteur_direct_2", "autre", "debiteur direct 1", "debiteur direct 2"],
                key=f"{prefix}_coh_t_{i}"
            )

            period = c2.selectbox(
                "Période",
                ["Annuel (€/an)", "Mensuel (€/mois)"],
                key=f"{prefix}_coh_rev_{i}_period"
            )

            if period.startswith("Annuel"):
                rev_annuel = c2.number_input(
                    "Revenus nets (€/an)",
                    min_value=0.0, value=0.0, step=100.0,
                    key=f"{prefix}_coh_rev_{i}_val_a"
                )
            else:
                rev_m = c2.number_input(
                    "Revenus nets (€/mois)",
                    min_value=0.0, value=0.0, step=50.0,
                    key=f"{prefix}_coh_rev_{i}_val_m"
                )
                rev_annuel = float(rev_m) * 12.0

            c2.caption(f"➡️ Retenu : {rev_annuel:.2f} €/an")

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
                "name": str(nom).strip(),
                "type": typ,
                "revenu_net_annuel": float(rev_annuel),
                "exclure": bool(excl),
                "date_quitte_menage": str(dq) if isinstance(dq, date) else None
            })

    answers["cohabitants_art34"] = cohabitants
    answers["pf_links"] = pf_links
    return answers

def ui_cohabitants_cascade(prefix: str) -> list[dict]:
    st.caption("Encode les cohabitants et coche leur rôle (pour les retrouver dans le paramétrage par dossier).")

    cohs = []
    nb = st.number_input("Nombre de cohabitants à encoder", min_value=0, value=3, step=1, key=f"{prefix}_nb")

    for i in range(int(nb)):
        st.markdown(f"**Cohabitant {i+1}**")
        c1, c2, c3 = st.columns([2.2, 1.2, 1])

        mid = c1.text_input("ID (court, unique) ex: X, Y, M1", value=f"M{i+1}", key=f"{prefix}_id_{i}")
        name = c1.text_input("Nom (optionnel)", value="", key=f"{prefix}_name_{i}")

        period = c2.selectbox("Période revenus", ["Annuel (€/an)", "Mensuel (€/mois)"], key=f"{prefix}_p_{i}")
        if period.startswith("Annuel"):
            rev_annuel = c2.number_input("Revenus nets (€/an)", min_value=0.0, value=0.0, step=100.0, key=f"{prefix}_ra_{i}")
        else:
            rev_m = c2.number_input("Revenus nets (€/mois)", min_value=0.0, value=0.0, step=50.0, key=f"{prefix}_rm_{i}")
            rev_annuel = float(rev_m) * 12.0
        c2.caption(f"➡️ Retenu : {rev_annuel:.2f} €/an")

        excl = c3.checkbox("Exclure", value=False, key=f"{prefix}_ex_{i}")

        st.markdown("**Rôle(s) pour la cascade**")
        r1, r2, r3 = st.columns(3)
        is_partenaire = r1.checkbox("Partenaire", value=False, key=f"{prefix}_part_{i}")
        is_deg1 = r2.checkbox("Débiteur 1er degré", value=False, key=f"{prefix}_d1_{i}")
        is_deg2 = r3.checkbox("Débiteur 2e degré", value=False, key=f"{prefix}_d2_{i}")
        is_art34 = c3.checkbox("Candidat art.34", value=True, key=f"{prefix}_a34_{i}")
        
        m = {
            "id": str(mid).strip(),
            "name": str(name).strip(),
            "revenu_net_annuel": float(rev_annuel),
            "exclure": bool(excl),
            "tag_partenaire": bool(is_partenaire),
            "tag_deg1": bool(is_deg1),
            "tag_deg2": bool(is_deg2),
            "art34_candidate": bool(is_art34),
            "_source": "cohabitant",
        }
        if m["id"]:
            cohs.append(m)

    return cohs
# jusqu'ici 
def annual_from_revenus_list(rev_list: list, cfg_soc: dict, cfg_ale: dict) -> float:
    return float(revenus_annuels_apres_exonerations(rev_list or [], cfg_soc, cfg_ale))

# ============================================================
# MODE DOSSIER (SINGLE / MULTI)
# ============================================================
#st.subheader("Mode dossier")
#multi_mode = st.checkbox("Plusieurs demandes RIS — comparer / calculer un ménage", value=False)
st.subheader("Choix du mode")

mode_global = st.radio(
    "Que veux-tu faire ?",
    ["Mode simple", "Mode Débiteurs#cascade"],
    index=0,
)

mode_cascade = (mode_global == "Mode Débiteurs#cascade")

# Dans les 2 cas, on peut avoir 1 ou plusieurs dossiers
multi_mode = st.checkbox("Plusieurs dossiers", value=False)

st.divider()
# ------------------------------------------------------------
# MODE MULTI
# ------------------------------------------------------------
#if multi_mode:
    #st.subheader("Choix du mode multi")
    #advanced_household = True
    #st.info("Mode multi : ménage avancé activé (cascade art.34).")
if multi_mode:
    if mode_cascade:
        st.subheader("Mode Débiteurs#cascade — plusieurs dossiers")
        advanced_household = True
        st.info("Tu es en mode Débiteurs#cascade.")
    else:
        st.subheader("Mode simple — plusieurs dossiers")
        advanced_household = False
        st.info("Tu es en mode simple.")
    #st.subheader("Choix du mode multi")
    #advanced_household = st.checkbox(
        #"Ménage avancé (cascade art.34 : priorité 1er degré -> 2e degré + pool)",
        #value=True
    #)

    nb_dem = st.number_input(
        "Nombre de dossiers/demandes à calculer",
        min_value=1, max_value=4, value=1, step=1
    )
    st.caption("Exemple : 1 dossier peut représenter un couple (2 personnes, 1 seul dossier).")

    st.subheader("A) Dossiers / demandes")
    dossiers = []

    for i in range(int(nb_dem)):
        st.markdown(f"### Dossier {i+1}")

        demandeur_nom = st.text_input("Nom du demandeur", value="", key=f"hd_dem_nom_{i}")
        label = st.text_input("Nom/Label", value=f"Dossier {i+1}", key=f"hd_lab_{i}")

        cat = st.selectbox(
            "Catégorie RIS",
            options=["cohab", "isole", "fam_charge"],
            format_func=cat_label,
            key=f"hd_cat_{i}"
        )

        enfants = st.number_input("Enfants à charge", min_value=0, value=0, step=1, key=f"hd_enf_{i}")
        d_dem = st.date_input("Date de demande", value=date.today(), key=f"hd_date_{i}")

        is_couple = st.checkbox("Dossier COUPLE (2 demandeurs ensemble)", value=False, key=f"hd_couple_{i}")
        demandeur2_nom = ""
        if is_couple:
            demandeur2_nom = st.text_input("Nom du demandeur 2 (conjoint)", value="", key=f"hd_dem2_nom_{i}")

        st.markdown("**Revenus nets (demandeur 1)**")
        rev1 = ui_revenus_block(f"hd_rev1_{i}")

        rev2 = []
        if is_couple:
            st.markdown("**Revenus nets (demandeur 2 / conjoint)**")
            rev2 = ui_revenus_block(f"hd_rev2_{i}")

        st.markdown("**PF à compter (spécifiques à CE dossier)**")
        pf_m = st.number_input(
            "PF à compter (€/mois)",
            min_value=0.0,
            value=float(cfg["pf"].get("pf_mensuel_defaut", 0.0)),
            step=10.0,
            key=f"hd_pf_{i}"
        )

        with st.expander("Patrimoine & ressources PERSONNELS (ce dossier)", expanded=False):
            pat_perso = ui_patrimoine_like_simple(prefix=f"hd_pat_perso_{i}")

        share_art34 = False
        if advanced_household:
            share_art34 = st.checkbox(
                "Enfant/Jeune demandeur : partager le pool art.34 (si plusieurs dossiers avec mêmes débiteurs 1er degré)",
                value=False,
                key=f"hd_share_{i}"
            )

        dossiers.append({
            "idx": i,
            "label": label,
            "demandeur_nom": str(demandeur_nom).strip(),
            "demandeur2_nom": str(demandeur2_nom).strip() if is_couple else "",
            "categorie": cat,
            "enfants_a_charge": int(enfants),
            "date_demande": d_dem,
            "couple_demandeur": bool(is_couple),
            "revenus_demandeur_annuels": rev1,
            "revenus_conjoint_annuels": rev2 if is_couple else [],
            "prestations_familiales_a_compter_mensuel": float(pf_m),
            "patrimoine_perso": pat_perso,
            "share_art34": bool(share_art34),
            "art34_deg1_ids": [],
            "art34_deg2_ids": [],
            "include_ris_from_dossiers": [],
        })

  
    if not advanced_household:
        with st.expander("Cohabitants admissibles (art.34) — ménage", expanded=False):
            menage_common = ui_menage_common(
                "hd_menage",
                nb_demandeurs=int(nb_dem),
                enable_pf_links=True,
                show_simple_art34=True
            )
    else:
        menage_common = {}  # en cascade, on ne passe pas par ce mode simple
      #with st.expander("Patrimoine & ressources du ménage (communes)", expanded=False):
        #pat_common_ui = ui_patrimoine_like_simple(prefix="hd_menage_pat_common")

    # ✅ on fusionne dans un seul dict “ménage commun”
    #menage_common = (menage_common or {})
    #menage_common.update(pat_common_ui or {})

    # PF-links -> dossiers
    #for link in menage_common.get("pf_links", []):
        #idx = int(link["dem_index"])
        #if 0 <= idx < len(dossiers):
            #dossiers[idx]["prestations_familiales_a_compter_mensuel"] += float(link["pf_mensuel"])
    menage_common = {}

    # Ménage avancé : membres & débiteurs
    household = {"members": [], "members_by_id": {}}
    if advanced_household:
        st.divider()
        st.subheader("C) Ménage avancé — Membres (IDs) & débiteurs art.34")
        st.caption("Préremplissage : les demandeurs sont ajoutés automatiquement (avec leurs revenus calculés).")

        prefill_demandeurs = st.checkbox("Préremplir les demandeurs", value=True, key="prefill_dem")

        members = []
        if prefill_demandeurs:
            for d in dossiers:
                id1 = f"D{d['idx']+1}A"
                name1 = (d.get("demandeur_nom") or "").strip() or f"Demandeur D{d['idx']+1}A"
                rev1_ann = annual_from_revenus_list(d.get("revenus_demandeur_annuels", []), cfg["socio_prof"], cfg["ale"])
                members.append({"id": id1, "name": name1, "revenu_net_annuel": float(rev1_ann), "exclure": False, "art34_candidate": True, "_source": "demandeur"})
                if bool(d.get("couple_demandeur", False)):
                    id2 = f"D{d['idx']+1}B"
                    name2 = (d.get("demandeur2_nom") or "").strip() or f"Demandeur D{d['idx']+1}B"
                    rev2_ann = annual_from_revenus_list(d.get("revenus_conjoint_annuels", []), cfg["socio_prof"], cfg["ale"])
                    members.append({"id": id2, "name": name2, "revenu_net_annuel": float(rev2_ann), "exclure": False, "art34_candidate": True, "_source": "demandeur"})

        nb_autres = st.number_input("Nombre d’AUTRES membres à encoder (hors demandeurs)", min_value=0, value=3, step=1, key="nb_autres_membres")
        for j in range(int(nb_autres)):
            st.markdown(f"**Autre membre {j+1}**")
            c1, c2, c3 = st.columns([2, 1, 1])
            is_art34 = c3.checkbox("Candidat art.34", value=True, key=f"mem_art34_{j}")
            mid = c1.text_input("ID court (ex: X, Y, E)", value=f"M{j+1}", key=f"mem_id_{j}")
            name = c1.text_input("Nom (optionnel)", value="", key=f"mem_name_{j}")
            #rev_annuel, _p = ui_money_period_input("Revenus nets", key_prefix=f"mem_rev_{j}", default=0.0, step=100.0)
            period = c2.selectbox(
                "Période",
                ["Annuel (€/an)", "Mensuel (€/mois)"],
                key=f"mem_rev_{j}_period"
            )

            if period.startswith("Annuel"):
                rev_annuel = c2.number_input(
                    "Revenus nets (€/an)",
                    min_value=0.0, value=0.0, step=100.0,
                    key=f"mem_rev_{j}_val_a"
                )
            else:
                rev_m = c2.number_input(
                    "Revenus nets (€/mois)",
                    min_value=0.0, value=0.0, step=50.0,
                    key=f"mem_rev_{j}_val_m"
                )
                rev_annuel = float(rev_m) * 12.0

            c2.caption(f"➡️ Retenu : {rev_annuel:.2f} €/an")

            excl = c3.checkbox("Exclure (équité)", value=False, key=f"mem_excl_{j}")
            m = {"id": str(mid).strip(), "name": str(name).strip(), "revenu_net_annuel": float(rev_annuel), "exclure": bool(excl), "art34_candidate": bool(is_art34), "_source": "autre"}
            
            if m["id"]:
                members.append(m)
        
        #members_by_id = {}
        #for m in members:
            #if m.get("exclure", False):
                #continue
            #if m.get("id"):
                #members_by_id[m["id"]] = m
        #household = {"members": members, "members_by_id": members_by_id}
        #ids_available = list(members_by_id.keys())

        #st.divider()

        #st.subheader("D) Paramétrage art.34 par dossier (cascade + injections RI)")
        #for d in dossiers:
            #st.markdown(f"### {d['label']} — art.34")
            #c1, c2 = st.columns(2)
            #d["art34_deg1_ids"] = c1.multiselect(
                #"Débiteurs 1er degré",
                #options=ids_available,
                #format_func=lambda mid: f"{mid} — {household['members_by_id'].get(mid, {}).get('name','')}".strip(" —"),
                #default=deg1_defaults,
                #key=f"d_{d['idx']}_deg1"
            #)

            #d["art34_deg2_ids"] = c2.multiselect(
                #"Débiteurs 2e degré (si 1er degré = 0)",
                #options=ids_available,
                #format_func=lambda mid: f"{mid} — {household['members_by_id'].get(mid, {}).get('name','')}".strip(" —"),
                #default=deg2_defaults,
                #key=f"d_{d['idx']}_deg2"
            #)
            
            #d["art34_deg1_ids"] = c1.multiselect(
                #"Débiteurs 1er degré (cohabitants débiteurs d'aliments)",
                #options=ids_available,
                #format_func=lambda mid: f"{mid} — {household['members_by_id'].get(mid, {}).get('name','')}".strip(" —"),
                #default=[],
                #key=f"d_{d['idx']}_deg1"
            #)
            #d["art34_deg2_ids"] = c2.multiselect(
                #"Débiteurs 2e degré (si 1er degré = 0)",
                #options=ids_available,
                #format_func=lambda mid: f"{mid} — {household['members_by_id'].get(mid, {}).get('name','')}".strip(" —"),
                #default=[],
                #key=f"d_{d['idx']}_deg2"
            #)
            #d["include_ris_from_dossiers"] = st.multiselect(
                #"Injection : ajouter le RI mensuel d’autres dossiers au groupe 1er degré (avant N×taux)",
                #options=[k for k in range(len(dossiers))],
                #format_func=lambda k: f"{k+1} — {dossiers[k]['label']}",
                #default=[],
                #key=f"d_{d['idx']}_risinj"
            #)

        # ... fin de construction members / members_by_id
        members_by_id = {}
        for m in members:
            if m.get("exclure", False):
                continue
            if m.get("id"):
                members_by_id[m["id"]] = m

        household = {"members": members, "members_by_id": members_by_id}
        #ids_available = list(members_by_id.keys())
        ids_all = list(members_by_id.keys())
        ids_art34 = [mid for mid in ids_all if bool(members_by_id[mid].get("art34_candidate", False))]

        # (optionnel) defaults si tu utilises des tags (sinon laisse vide)
        deg1_defaults = [mid for mid in ids_all if members_by_id.get(mid, {}).get("tag_deg1")]
        deg2_defaults = [mid for mid in ids_all if members_by_id.get(mid, {}).get("tag_deg2")]
        #deg1_defaults = [mid for mid in ids_all if members_by_id[mid].get("tag_deg1")]
        #deg2_defaults = [mid for mid in ids_all if members_by_id[mid].get("tag_deg2")]

        st.divider()
        st.subheader("D) Paramétrage art.34 par dossier (cascade + injections RI)")

        for d in dossiers:
            st.markdown(f"### {d['label']} — art.34")
            c1, c2 = st.columns(2)

            members_by_id = household.get("members_by_id", {})  # normalement déjà présent
            ids_all = list(members_by_id.keys())

             # Pool complet (demandeurs + autres), en excluant seulement ceux "exclude"
            ids_available = [mid for mid in ids_all if not bool(members_by_id.get(mid, {}).get("exclure", False))]
            ids_art34 = [mid for mid in ids_available if bool(members_by_id.get(mid, {}).get("art34_candidate", False))]
            
            d["art34_deg1_ids"] = c1.multiselect(
                "Débiteurs 1er degré",
                options=ids_art34,
                format_func=lambda mid: f"{mid} — {household['members_by_id'].get(mid, {}).get('name','')}".strip(" —"),
                default=deg1_defaults,   # ou [] si tu ne veux pas de préselection
                key=f"d_{d['idx']}_deg1"
            )

            d["art34_deg2_ids"] = c2.multiselect(
                "Débiteurs 2e degré (si 1er degré = 0)",
                options=ids_art34,
                format_func=lambda mid: f"{mid} — {household['members_by_id'].get(mid, {}).get('name','')}".strip(" —"),
                default=deg2_defaults,   # ou []
                key=f"d_{d['idx']}_deg2"
            )
            
            # ✅ nettoyage si un membre a été sélectionné puis décoché "candidat art.34"
            d["art34_deg1_ids"] = [mid for mid in d.get("art34_deg1_ids", []) if mid in ids_art34]
            d["art34_deg2_ids"] = [mid for mid in d.get("art34_deg2_ids", []) if mid in ids_art34]


            
            d["include_ris_from_dossiers"] = st.multiselect(
                "Injection : ajouter le RI mensuel d’autres dossiers au groupe 1er degré (avant N×taux)",
                options=[k for k in range(len(dossiers))],
                format_func=lambda k: f"{k+1} — {dossiers[k]['label']}",
                default=[],
                key=f"d_{d['idx']}_risinj"
            )

    # CALCUL MULTI
    st.divider()
    if st.button("Calculer (multi)"):
        taux_art34 = float(cfg["art34"]["taux_a_laisser_mensuel"])

        # partage : par groupe d'IDs (deg1), si plusieurs dossiers "share_art34"
        share_plan = {}
        if advanced_household:
            for d in dossiers:
                if not d.get("share_art34", False):
                    continue
                ids = list(d.get("art34_deg1_ids", []) or [])
                if not ids:
                    continue
                key = make_pool_key(ids)
                share_plan.setdefault(key, {"count": 0, "per": 0.0})
                share_plan[key]["count"] += 1

        pools = {}         # pool restant par groupe
        prior_results = [] # résultats précédents (pour injections RI)
        results = []
        pdf_buffers = []

        for d in dossiers:
            # answers = ménage commun + dossier
            #answers = {}
            #answers.update(menage_common or {})
            # ✅ Patrimoine commun vs perso
            #answers["_patrimoine_common"] = _extract_patrimoine(menage_common or {})
            #answers["_patrimoine_perso"]  = _extract_patrimoine(d.get("patrimoine_perso") or {})
            answers = {}
            answers["_patrimoine_common"] = _extract_patrimoine({})  # plus de commun
            answers["_patrimoine_perso"]  = _extract_patrimoine(d.get("patrimoine_perso") or {})

            answers.update({
                "categorie": d["categorie"],
                "enfants_a_charge": int(d["enfants_a_charge"]),
                "date_demande": d["date_demande"],
                "couple_demandeur": bool(d["couple_demandeur"]),
                "demandeur_nom": d.get("demandeur_nom", ""),
                "revenus_demandeur_annuels": d.get("revenus_demandeur_annuels", []),
                "revenus_conjoint_annuels": d.get("revenus_conjoint_annuels", []),
                "prestations_familiales_a_compter_mensuel": float(d.get("prestations_familiales_a_compter_mensuel", 0.0)),
            })

            # pour le PDF, en mode avancé, on n'encode pas "cohabitants_art34" en simple
            if advanced_household:
                answers["cohabitants_art34"] = []  # on laisse vide, le PDF utilise debug_* cascade

            # calcul standard (segments + mois suivants)
            seg_first = compute_first_month_segments(answers, engine)
            res_ms = seg_first.get("detail_mois_suivants", {}) or compute_officiel_cpas_annuel(answers, engine)

            # ✅ Appliquer la CASCADE art.34 (ménage avancé)
            if advanced_household:
                
                besoin_m = float(res_ms.get("ris_theorique_mensuel", 0.0))  # ✅ gap mensuel avant art.34

                art34_adv = compute_art34_menage_avance_cascade(
                    dossier=d,
                    household=household,
                    taux=taux_art34,
                    pools=pools,
                    share_plan=share_plan,
                    prior_results=prior_results,
                    besoin_m=besoin_m
                )

                # override mois suivants
                res_ms["art34_mode"] = art34_adv.get("art34_mode", "MENAGE_AVANCE_CASCADE")
                res_ms["art34_degree_utilise"] = art34_adv.get("art34_degree_utilise", 0)
                res_ms["cohabitants_part_a_compter_mensuel"] = float(art34_adv.get("cohabitants_part_a_compter_mensuel", 0.0))
                res_ms["cohabitants_part_a_compter_annuel"] = float(art34_adv.get("cohabitants_part_a_compter_annuel", 0.0))
                res_ms["debug_deg1"] = art34_adv.get("debug_deg1")
                res_ms["debug_deg2"] = art34_adv.get("debug_deg2")
                res_ms["ris_injecte_mensuel"] = art34_adv.get("ris_injecte_mensuel", 0.0)
                res_ms["taux_a_laisser_mensuel"] = float(res_ms.get("taux_a_laisser_mensuel", taux_art34))
                


                # recalcul totaux + RI
                total_dem = float(res_ms.get("total_ressources_demandeur_avant_immunisation_annuel", 0.0))
                total_coh = float(res_ms["cohabitants_part_a_compter_annuel"])
                total_av = r2(total_dem + total_coh)

                taux_ann = float(res_ms.get("taux_ris_annuel", 0.0))
                immu = float(res_ms.get("immunisation_simple_annuelle", 0.0))
                total_ap = r2(max(0.0, total_av - immu))
                ri_ann = r2(max(0.0, taux_ann - total_ap) if taux_ann > 0 else 0.0)

                res_ms["total_ressources_cohabitants_annuel"] = float(total_coh)
                res_ms["total_ressources_avant_immunisation_simple_annuel"] = float(total_av)
                res_ms["total_ressources_apres_immunisation_simple_annuel"] = float(total_ap)
                res_ms["ris_theorique_annuel"] = float(ri_ann)
                res_ms["ris_theorique_mensuel"] = float(r2(ri_ann / 12.0))

                # override segments 1er mois (cohérence + pdf)
                if seg_first and seg_first.get("segments"):
                    for s in seg_first["segments"]:
                        res_seg = s.get("_detail_res")
                        if not isinstance(res_seg, dict):
                            continue

                        res_seg["art34_mode"] = res_ms["art34_mode"]
                        res_seg["art34_degree_utilise"] = res_ms["art34_degree_utilise"]
                        res_seg["debug_deg1"] = res_ms.get("debug_deg1")
                        res_seg["debug_deg2"] = res_ms.get("debug_deg2")
                        res_seg["ris_injecte_mensuel"] = res_ms.get("ris_injecte_mensuel", 0.0)

                        res_seg["cohabitants_part_a_compter_mensuel"] = res_ms["cohabitants_part_a_compter_mensuel"]
                        res_seg["cohabitants_part_a_compter_annuel"] = res_ms["cohabitants_part_a_compter_annuel"]
                        res_seg["total_ressources_cohabitants_annuel"] = res_ms["total_ressources_cohabitants_annuel"]

                        total_dem_seg = float(res_seg.get("total_ressources_demandeur_avant_immunisation_annuel", 0.0))
                        total_av_seg = r2(total_dem_seg + float(res_seg["total_ressources_cohabitants_annuel"]))
                        immu_seg = float(res_seg.get("immunisation_simple_annuelle", 0.0))
                        total_ap_seg = r2(max(0.0, total_av_seg - immu_seg))
                        taux_ann_seg = float(res_seg.get("taux_ris_annuel", 0.0))
                        ri_ann_seg = r2(max(0.0, taux_ann_seg - total_ap_seg) if taux_ann_seg > 0 else 0.0)
                        ri_m_seg = r2(ri_ann_seg / 12.0)

                        res_seg["total_ressources_avant_immunisation_simple_annuel"] = float(total_av_seg)
                        res_seg["total_ressources_apres_immunisation_simple_annuel"] = float(total_ap_seg)
                        res_seg["ris_theorique_annuel"] = float(ri_ann_seg)
                        res_seg["ris_theorique_mensuel"] = float(ri_m_seg)

                        s["ris_mensuel"] = float(ri_m_seg)
                        s["montant_segment"] = float(r2(float(ri_m_seg) * float(s.get("prorata", 0.0))))

                    seg_first["ris_1er_mois_total"] = float(r2(sum(float(s.get("montant_segment", 0.0)) for s in seg_first["segments"])))
                    seg_first["ris_mois_suivants"] = float(res_ms["ris_theorique_mensuel"])
                    seg_first["detail_mois_suivants"] = res_ms

            # PDF
            pdf_buf = make_decision_pdf_cpas(
                dossier_label=d.get("label", f"Dossier {d['idx']+1}"),
                answers_snapshot=answers,
                res_mois_suivants=res_ms,
                seg_first_month=seg_first,
                logo_path="logo.png",
                cfg_soc=cfg["socio_prof"],
                cfg_ale=cfg["ale"],
                cfg_cap=cfg["capital_mobilier"],
                cfg_immo=cfg["immo"],
                cfg_cession=cfg["cession"],
            )

            prior_results.append(res_ms)
            results.append({"dossier": d, "res": res_ms, "seg": seg_first})
            pdf_buffers.append(pdf_buf)

        # affichage résultats
        st.subheader("Résultats")
        for i, r in enumerate(results):
            d = r["dossier"]
            res = r["res"]
            seg = r["seg"]

            st.markdown(f"### {d['label']}")
            if d.get("demandeur_nom"):
                st.caption(f"Demandeur : {d['demandeur_nom']}")

            st.write(f"**RI mois suivant (référence)** : {euro(res.get('ris_theorique_mensuel',0))} € / mois")
            if seg and seg.get("segments"):
                st.write(f"**Total 1er mois** : {euro(seg.get('ris_1er_mois_total',0))} €")

            if advanced_household:
                st.caption(
                    f"Art.34 cascade — degré utilisé : {res.get('art34_degree_utilise',0)} | "
                    f"pris en compte (mensuel) : {euro(res.get('cohabitants_part_a_compter_mensuel',0))} €"
                )

            if pdf_buffers[i] is not None:
                st.download_button(
                    "Télécharger PDF (décision)",
                    data=pdf_buffers[i].getvalue(),
                    file_name=f"decision_{d['label'].replace(' ','_')}.pdf",
                    mime="application/pdf",
                    key=f"dl_pdf_{i}"
                )

else:
    st.subheader("Mode SIMPLE (single dossier)")
    advanced_single = mode_cascade

    # Choix (facultatif) : si tu veux aussi offrir le ménage avancé en single
    #advanced_single = st.checkbox(
        #"Activer ménage avancé (cascade art.34) en single",
        #value=False
    #)

    # --- Dossier ---
    st.markdown("### A) Demande")
    demandeur_nom = st.text_input("Nom du demandeur", value="", key="s_dem_nom")
    cat = st.selectbox(
        "Catégorie RIS",
        options=["cohab", "isole", "fam_charge"],
        format_func=cat_label,
        key="s_cat"
    )
    enfants = st.number_input("Enfants à charge", min_value=0, value=0, step=1, key="s_enf")
    d_dem = st.date_input("Date de demande", value=date.today(), key="s_date")

    is_couple = st.checkbox("Dossier COUPLE (2 demandeurs ensemble)", value=False, key="s_couple")
    demandeur2_nom = ""
    if is_couple:
        demandeur2_nom = st.text_input("Nom du demandeur 2 (conjoint)", value="", key="s_dem2_nom")

    st.markdown("**Revenus nets (demandeur 1)**")
    rev1 = ui_revenus_block("s_rev1")

    rev2 = []
    if is_couple:
        st.markdown("**Revenus nets (demandeur 2 / conjoint)**")
        rev2 = ui_revenus_block("s_rev2")

    st.markdown("**PF à compter (pour ce dossier)**")
    pf_m = st.number_input(
        "PF à compter (€/mois)",
        min_value=0.0,
        value=float(cfg["pf"].get("pf_mensuel_defaut", 0.0)),
        step=10.0,
        key="s_pf"
    )

    # --- Ménage commun (patrimoine + (art.34 simple si pas avancé)) ---
    with st.expander("B) Ménage / Patrimoine (mode simple)", expanded=True):
        menage_common = ui_menage_common(
            prefix="s_menage",
            nb_demandeurs=1,
            enable_pf_links=False,                 # en single, pas besoin de liens PF vers plusieurs dossiers
            show_simple_art34=not advanced_single  # ✅ si ménage avancé, on cache l’art.34 simple
        )

    # --- Ménage avancé en single (optionnel) ---
    # On garde ta logique cascade/pool, mais en single ça revient à calculer une fois.
    household = {"members": [], "members_by_id": {}}
    d_adv = {
        "idx": 0,
        "label": "Dossier",
        "share_art34": False,
        "art34_deg1_ids": [],
        "art34_deg2_ids": [],
        "include_ris_from_dossiers": [],
    }

    if advanced_single:
        st.divider()
        st.subheader("C) Ménage avancé — Membres (IDs) & débiteurs art.34 (single)")

        prefill = st.checkbox("Préremplir le(s) demandeur(s) comme membre(s)", value=True, key="s_prefill")

        members = []
        if prefill:
            # Demandeur A
            id1 = "A"
            name1 = demandeur_nom.strip() or "Demandeur A"
            rev1_ann_calc = annual_from_revenus_list(rev1, cfg["socio_prof"], cfg["ale"])
            members.append({"id": id1, "name": name1, "revenu_net_annuel": float(rev1_ann_calc), "exclure": False, "_source": "demandeur", "art34_candidate": True})
            # Conjoint B (si couple)
            if is_couple:
                id2 = "B"
                name2 = demandeur2_nom.strip() or "Demandeur B"
                rev2_ann_calc = annual_from_revenus_list(rev2, cfg["socio_prof"], cfg["ale"])
                members.append({"id": id2, "name": name2, "revenu_net_annuel": float(rev2_ann_calc), "exclure": False, "_source": "demandeur", "art34_candidate": True})

        #nb_autres = st.number_input("Nombre d’AUTRES membres à encoder (hors demandeurs)", min_value=0, value=2, step=1, key="s_nb_autres")
        #for j in range(int(nb_autres)):
            #st.markdown(f"**Autre membre {j+1}**")
            #c1, c2, c3 = st.columns([2, 1, 1])
            #mid = c1.text_input("ID court (ex: X, Y, E)", value=f"M{j+1}", key=f"s_mem_id_{j}")
            #name = c1.text_input("Nom (optionnel)", value="", key=f"s_mem_name_{j}")
            #rev_annuel, _p = ui_money_period_input("Revenus nets", key_prefix=f"s_mem_rev_{j}", default=0.0, step=100.0)
            #excl = c3.checkbox("Exclure (équité)", value=False, key=f"s_mem_excl_{j}")
            #m = {"id": str(mid).strip(), "name": str(name).strip(), "revenu_net_annuel": float(rev_annuel), "exclure": bool(excl), "_source": "autre"}
            #if m["id"]:
                #members.append(m)
            # Cohabitants encodés + tags débiteurs
        with st.expander("B) Cohabitants (mode Débiteurs#cascade)", expanded=True):
            coh_members = ui_cohabitants_cascade(prefix="hd_coh")

        members = []

        # Préremplissage demandeurs (comme avant)
        prefill_demandeurs = st.checkbox("Préremplir les demandeurs", value=True, key="prefill_dem")
        if prefill_demandeurs:
            for d in dossiers:
                id1 = f"D{d['idx']+1}A"
                name1 = (d.get("demandeur_nom") or "").strip() or f"Demandeur D{d['idx']+1}A"
                rev1_ann = annual_from_revenus_list(d.get("revenus_demandeur_annuels", []), cfg["socio_prof"], cfg["ale"])
                #members.append({"id": id1, "name": name1, "revenu_net_annuel": float(rev1_ann), "exclure": False, "_source": "demandeur",
                                #"tag_partenaire": False, "tag_deg1": False, "tag_deg2": False})
                #if bool(d.get("couple_demandeur", False)):
                    #id2 = f"D{d['idx']+1}B"
                    #name2 = (d.get("demandeur2_nom") or "").strip() or f"Demandeur D{d['idx']+1}B"
                    #rev2_ann = annual_from_revenus_list(d.get("revenus_conjoint_annuels", []), cfg["socio_prof"], cfg["ale"])
                    #members.append({"id": id2, "name": name2, "revenu_net_annuel": float(rev2_ann), "exclure": False, "_source": "demandeur",
                                    #"tag_partenaire": False, "tag_deg1": False, "tag_deg2": False})


                #members.append({
                members.append({
                    "id": id1,
                    "name": name1,
                    "revenu_net_annuel": float(rev1_ann),
                    "exclude": False,
                    "_source": "demandeur",
                    "role": "demandeur",
                    "art34_candidate": True,   # ✅ permet de le proposer comme débiteur/candidat
                    "tag_partenaire": False,
                    "tag_deg1": False,
                    "tag_deg2": False,
                })

                if bool(d.get("couple_demandeur", False)):
                    members.append({
                        "id": id2,
                        "name": name2,
                        "revenu_net_annuel": float(rev2_ann),
                        "exclude": False,
                        "_source": "demandeur",
                        "role": "demandeur",
                        "art34_candidate": True,  # ✅ idem
                        "tag_partenaire": False,
                        "tag_deg1": False,
                        "tag_deg2": False,
                    })

        # Ajout cohabitants encodés
        members.extend(coh_members or [])

        # Build members_by_id
        members_by_id = {}
        for m in members:
            if m.get("exclure", False):
                continue
            if m.get("id"):
                members_by_id[m["id"]] = m

        household = {"members": members, "members_by_id": members_by_id}
        ids_available = list(members_by_id.keys())

        # Listes filtrées (pour defaults)
        deg1_defaults = [mid for mid in ids_available if members_by_id[mid].get("tag_deg1")]
        deg2_defaults = [mid for mid in ids_available if members_by_id[mid].get("tag_deg2")]

        members_by_id = {}
        for m in members:
            if m.get("exclure", False):
                continue
            if m.get("id"):
                members_by_id[m["id"]] = m
        household = {"members": members, "members_by_id": members_by_id}
        ids_available = list(members_by_id.keys())

        st.divider()
        st.subheader("D) Paramétrage art.34 (cascade)")

        c1, c2 = st.columns(2)
        d_adv["art34_deg1_ids"] = c1.multiselect(
            "Débiteurs 1er degré",
            options=ids_available,
            format_func=lambda mid: f"{mid} — {household['members_by_id'].get(mid, {}).get('name','')}".strip(" —"),
            default=[],
            key="s_deg1"
        )
        d_adv["art34_deg2_ids"] = c2.multiselect(
            "Débiteurs 2e degré (si 1er degré = 0)",
            options=ids_available,
            format_func=lambda mid: f"{mid} — {household['members_by_id'].get(mid, {}).get('name','')}".strip(" —"),
            default=[],
            key="s_deg2"
        )

    # --- Calcul single ---
    st.divider()
    
    with st.expander("Patrimoine & ressources PERSONNELS (ce dossier) — fin de dossier", expanded=False):
        pat_perso_single = ui_patrimoine_like_simple(prefix="s_pat_perso")

    if st.button("Calculer (single)"):
        # answers = ménage commun + dossier
        answers = {}
        answers.update(menage_common or {})
        answers["_patrimoine_common"] = _extract_patrimoine(menage_common or {})
        answers["_patrimoine_perso"]  = _extract_patrimoine(pat_perso_single or {})  # ✅


        answers.update({
            "categorie": cat,
            "enfants_a_charge": int(enfants),
            "date_demande": d_dem,
            "couple_demandeur": bool(is_couple),
            "demandeur_nom": str(demandeur_nom).strip(),
            "revenus_demandeur_annuels": rev1,
            "revenus_conjoint_annuels": rev2 if is_couple else [],
            "prestations_familiales_a_compter_mensuel": float(pf_m),

        })

        # En ménage avancé: on ne veut pas l’art.34 simple
        if advanced_single:
            answers["cohabitants_art34"] = []

        seg_first = compute_first_month_segments(answers, engine)
        res_ms = seg_first.get("detail_mois_suivants", {}) or compute_officiel_cpas_annuel(answers, engine)

        # appliquer cascade si activée
        if advanced_single:
            taux_art34 = float(cfg["art34"]["taux_a_laisser_mensuel"])
            pools = {}
            share_plan = {}      # single -> inutile, mais gardé
            prior_results = []   # single -> vide (pas d’injection depuis autres dossiers)

            besoin_m = float(res_ms.get("ris_theorique_mensuel", 0.0))

            art34_adv = compute_art34_menage_avance_cascade(
                dossier=d_adv,
                household=household,
                taux=taux_art34,
                pools=pools,
                share_plan=share_plan,
                prior_results=prior_results,
                besoin_m=besoin_m
            )

            # override art34
            res_ms["art34_mode"] = art34_adv.get("art34_mode", "MENAGE_AVANCE_CASCADE")
            res_ms["art34_degree_utilise"] = art34_adv.get("art34_degree_utilise", 0)
            res_ms["cohabitants_part_a_compter_mensuel"] = float(art34_adv.get("cohabitants_part_a_compter_mensuel", 0.0))
            res_ms["cohabitants_part_a_compter_annuel"] = float(art34_adv.get("cohabitants_part_a_compter_annuel", 0.0))
            res_ms["debug_deg1"] = art34_adv.get("debug_deg1")
            res_ms["debug_deg2"] = art34_adv.get("debug_deg2")
            res_ms["ris_injecte_mensuel"] = art34_adv.get("ris_injecte_mensuel", 0.0)
            res_ms["taux_a_laisser_mensuel"] = art34_adv.get("taux_a_laisser_mensuel", taux_art34)


            # recalcul totaux + RI
            total_dem = float(res_ms.get("total_ressources_demandeur_avant_immunisation_annuel", 0.0))
            total_coh = float(res_ms["cohabitants_part_a_compter_annuel"])
            total_av = r2(total_dem + total_coh)

            taux_ann = float(res_ms.get("taux_ris_annuel", 0.0))
            immu = float(res_ms.get("immunisation_simple_annuelle", 0.0))
            total_ap = r2(max(0.0, total_av - immu))
            ri_ann = r2(max(0.0, taux_ann - total_ap) if taux_ann > 0 else 0.0)

            res_ms["total_ressources_cohabitants_annuel"] = float(total_coh)
            res_ms["total_ressources_avant_immunisation_simple_annuel"] = float(total_av)
            res_ms["total_ressources_apres_immunisation_simple_annuel"] = float(total_ap)
            res_ms["ris_theorique_annuel"] = float(ri_ann)
            res_ms["ris_theorique_mensuel"] = float(r2(ri_ann / 12.0))

            # segments cohérents
            if seg_first and seg_first.get("segments"):
                for s in seg_first["segments"]:
                    res_seg = s.get("_detail_res")
                    if not isinstance(res_seg, dict):
                        continue

                    res_seg["art34_mode"] = res_ms["art34_mode"]
                    res_seg["art34_degree_utilise"] = res_ms["art34_degree_utilise"]
                    res_seg["debug_deg1"] = res_ms.get("debug_deg1")
                    res_seg["debug_deg2"] = res_ms.get("debug_deg2")
                    res_seg["ris_injecte_mensuel"] = res_ms.get("ris_injecte_mensuel", 0.0)

                    res_seg["cohabitants_part_a_compter_mensuel"] = res_ms["cohabitants_part_a_compter_mensuel"]
                    res_seg["cohabitants_part_a_compter_annuel"] = res_ms["cohabitants_part_a_compter_annuel"]
                    res_seg["total_ressources_cohabitants_annuel"] = res_ms["total_ressources_cohabitants_annuel"]

                    total_dem_seg = float(res_seg.get("total_ressources_demandeur_avant_immunisation_annuel", 0.0))
                    total_av_seg = r2(total_dem_seg + float(res_seg["total_ressources_cohabitants_annuel"]))
                    immu_seg = float(res_seg.get("immunisation_simple_annuelle", 0.0))
                    total_ap_seg = r2(max(0.0, total_av_seg - immu_seg))
                    taux_ann_seg = float(res_seg.get("taux_ris_annuel", 0.0))
                    ri_ann_seg = r2(max(0.0, taux_ann_seg - total_ap_seg) if taux_ann_seg > 0 else 0.0)
                    ri_m_seg = r2(ri_ann_seg / 12.0)

                    res_seg["total_ressources_avant_immunisation_simple_annuel"] = float(total_av_seg)
                    res_seg["total_ressources_apres_immunisation_simple_annuel"] = float(total_ap_seg)
                    res_seg["ris_theorique_annuel"] = float(ri_ann_seg)
                    res_seg["ris_theorique_mensuel"] = float(ri_m_seg)

                    s["ris_mensuel"] = float(ri_m_seg)
                    s["montant_segment"] = float(r2(float(ri_m_seg) * float(s.get("prorata", 0.0))))

                seg_first["ris_1er_mois_total"] = float(r2(sum(float(s.get("montant_segment", 0.0)) for s in seg_first["segments"])))
                seg_first["ris_mois_suivants"] = float(res_ms["ris_theorique_mensuel"])
                seg_first["detail_mois_suivants"] = res_ms

        # PDF
        pdf_buf = make_decision_pdf_cpas(
            dossier_label="Dossier",
            answers_snapshot=answers,
            res_mois_suivants=res_ms,
            seg_first_month=seg_first,
            logo_path="logo.png",
            cfg_soc=cfg["socio_prof"],
            cfg_ale=cfg["ale"],
            cfg_cap=cfg["capital_mobilier"],
            cfg_immo=cfg["immo"],
            cfg_cession=cfg["cession"],
        )

        # Affichage
        st.subheader("Résultat")
        if demandeur_nom.strip():
            st.caption(f"Demandeur : {demandeur_nom.strip()}")

        st.write(f"**RI mois suivant (référence)** : {euro(res_ms.get('ris_theorique_mensuel',0))} € / mois")
        if seg_first and seg_first.get("segments"):
            st.write(f"**Total 1er mois** : {euro(seg_first.get('ris_1er_mois_total',0))} €")

        if advanced_single:
            st.caption(
                f"Art.34 cascade — degré utilisé : {res_ms.get('art34_degree_utilise',0)} | "
                f"pris en compte (mensuel) : {euro(res_ms.get('cohabitants_part_a_compter_mensuel',0))} €"
            )

        if pdf_buf is not None:
            st.download_button(
                "Télécharger PDF (décision)",
                data=pdf_buf.getvalue(),
                file_name="decision_dossier.pdf",
                mime="application/pdf",
                key="dl_pdf_single"
            )
