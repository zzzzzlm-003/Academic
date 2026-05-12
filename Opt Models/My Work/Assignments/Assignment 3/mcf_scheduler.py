"""
MCF Monthly Van Scheduling - Optimization Model
IEOR 4004 Assignment 3
Uses Gurobi if available; falls back to PuLP/CBC otherwise.

Run: python mcf_scheduler.py
     python mcf_scheduler.py eda
     python mcf_scheduler.py --gurobi   # force Gurobi
     python mcf_scheduler.py --pulp     # force PuLP

If you have Gurobi in conda, use: conda activate <env> && python mcf_scheduler.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    from pulp import LpMaximize, LpProblem, LpVariable, LpInteger, lpSum, PULP_CBC_CMD

# ============== DATA LOADING ==============
DATA_PATH = "IEOR4004_HW3_MobileCareData.xlsx"
# Patient pool: all patients from Case_PatientList (DUE BACK filtering optional)
SEVERITY_MAP = {"MildIntermittent": 1, "MildPersistent": 2, "ModeratePersistent": 3, "SeverePersistent": 4}

def load_data():
    xl = pd.ExcelFile(DATA_PATH)
    
    # School locations
    loc = pd.read_excel(xl, "School Locations", header=None)
    loc = loc.iloc[10:30].copy()
    loc.columns = ["School", "X", "Y"]
    loc["School"] = loc["School"].str.extract(r"(\d+)").astype(int)
    loc["X"] = pd.to_numeric(loc["X"], errors="coerce")
    loc["Y"] = pd.to_numeric(loc["Y"], errors="coerce")
    loc = loc.dropna()
    
    # Patient list: school -> patients
    pl = pd.read_excel(xl, "Case_PatientList")
    patient_to_school = {}
    for col in pl.columns:
        sch = int(col.replace("SCHOOL #", ""))
        for p in pl[col].dropna():
            pid = int(str(p).replace("Patient #", ""))
            patient_to_school[pid] = sch
    
    # Patient severity
    sev = pd.read_excel(xl, "Case_PatientSeverity")
    sev["pid"] = sev["PATIENT"].str.extract(r"(\d+)").astype(int)
    sev["severity"] = sev["SEVERITY"].map(SEVERITY_MAP)
    patient_severity = dict(zip(sev["pid"], sev["severity"]))
    
    return xl, loc, patient_to_school, patient_severity


def compute_distance_params(loc):
    """Rectilinear distance: 0.1 * (|Δx| + |Δy|) miles per data sheet.
    Returns d_s: distance from each school to centroid (proxy for route length)."""
    loc = loc.set_index("School")
    cx = loc["X"].mean()
    cy = loc["Y"].mean()
    d_s = {}
    for s in loc.index:
        dx = abs(loc.loc[s, "X"] - cx)
        dy = abs(loc.loc[s, "Y"] - cy)
        d_s[s] = 0.1 * (dx + dy)  # miles
    return d_s


def extract_patient_info_from_apps(xl):
    """Extract latest appointment info per patient from APPS sheets."""
    records = []
    for k in range(20):
        try:
            df = pd.read_excel(xl, f"Case_SCH#{k}_APPS")
            df["school"] = k
            records.append(df)
        except Exception:
            pass
    apps = pd.concat(records, ignore_index=True)
    
    # Parse dates
    apps["APP DATE"] = pd.to_datetime(apps["APP DATE"], errors="coerce")
    apps["LAST SEEN"] = pd.to_datetime(apps["LAST SEEN"], errors="coerce")
    apps["DUE BACK"] = pd.to_datetime(apps["DUE BACK"], errors="coerce")
    apps["LAST DUE BACK"] = pd.to_datetime(apps["LAST DUE BACK"], errors="coerce")
    apps["pid"] = apps["PATIENT"].astype(str).str.extract(r"(\d+)").astype(float).fillna(-1).astype(int)
    
    # Numeric HOSP, ER, DAYS
    apps["HOSP"] = pd.to_numeric(apps["HOSP."], errors="coerce").fillna(0)
    apps["ER"] = pd.to_numeric(apps["ER-VISIT"], errors="coerce").fillna(0)
    apps["DAYS_MISSED"] = pd.to_numeric(apps["DAYS MISSED"], errors="coerce").fillna(0)
    apps["NO_SHOW"] = (apps["CURRENT DIAG"] == "NO SHOW").astype(int)
    
    # Latest record per patient (take most recent APP DATE)
    apps = apps.sort_values("APP DATE", ascending=False)
    latest = apps.groupby("pid").first().reset_index()
    latest["school"] = latest.apply(lambda r: r.get("school", 0), axis=1)
    
    return latest

def is_elevated_risk(row):
    if not row or (isinstance(row, float) and pd.isna(row)):
        return False
    r = row
    hosp = r.get("HOSP", 0) or 0
    er = r.get("ER", 0) or 0
    days = r.get("DAYS_MISSED", 0) or 0
    last_seen = r.get("LAST SEEN")
    last_diag = r.get("LAST DIAG", "")
    if hosp > 0 or er > 0:
        return True
    if days > 2:
        return True
    if pd.notna(last_seen):
        ref = pd.Timestamp(2012, 2, 1)
        if (ref - last_seen).days > 90:
            return True
    if last_diag in ["Worsened", "Unchanged"]:
        return True
    return False

def build_patient_pool(xl, patient_to_school, patient_severity):
    """Build p[s][l][r] and psi[s][l][r] (NO SHOW fraction).
    Pool = all patients from Case_PatientList. Risk/NO SHOW from APPS.
    """
    latest = extract_patient_info_from_apps(xl)
    apps_by_pid = latest.set_index("pid").to_dict("index") if len(latest) > 0 else {}
    
    p = {}
    psi_num = {}
    psi_den = {}
    patient_details = []
    
    for s in range(20):
        p[s] = {(l, r): 0 for l in range(1, 5) for r in [0, 1]}
        psi_num[s] = {(l, r): 0 for l in range(1, 5) for r in [0, 1]}
        psi_den[s] = {(l, r): 0 for l in range(1, 5) for r in [0, 1]}
    
    for pid, school in patient_to_school.items():
        sev = patient_severity.get(pid)
        if sev is None:
            continue
        row = apps_by_pid.get(pid, {})
        elev = 1 if is_elevated_risk(row) else 0
        p[school][(sev, elev)] += 1
        psi_den[school][(sev, elev)] += 1
        if row.get("NO_SHOW", 0) == 1:
            psi_num[school][(sev, elev)] += 1
        patient_details.append({
            "pid": pid, "school": school, "severity": sev, "elevated": elev,
            "no_show": row.get("NO_SHOW", 0)
        })
    
    psi = {}
    for s in range(20):
        psi[s] = {}
        for (l, r) in p[s]:
            d = psi_den[s][(l, r)]
            n = psi_num[s][(l, r)]
            psi[s][(l, r)] = n / d if d > 0 else 0
    
    return p, psi, patient_details

def infer_capacities(xl):
    """Infer van capacity and school capacity from historical APPS."""
    apps_counts = []
    for k in range(20):
        try:
            df = pd.read_excel(xl, f"Case_SCH#{k}_APPS")
            df["APP DATE"] = pd.to_datetime(df["APP DATE"], errors="coerce")
            df = df.dropna(subset=["APP DATE"])
            per_day = df.groupby([df["APP DATE"].dt.date, "VAN"]).size()
            apps_counts.extend(per_day.tolist())
        except Exception:
            pass
    if apps_counts:
        avg_per_van_day = np.mean(apps_counts)
        # ~20 weekdays per month
        c_van = int(np.ceil(avg_per_van_day * 20)) if avg_per_van_day > 0 else 150
    else:
        c_van = 150
    c_van = min(c_van, 300)
    a_school = 200  # max appointments per school per month (from historical capacity)
    return c_van, a_school

def solve_mcf(use_gurobi=None):
    """Solve MCF scheduling. use_gurobi=True forces Gurobi; False forces PuLP; None=auto."""
    xl, loc, patient_to_school, patient_severity = load_data()
    p, psi, patient_details = build_patient_pool(xl, patient_to_school, patient_severity)
    c_van, a_school = infer_capacities(xl)
    d_s = compute_distance_params(loc)  # distance from school to centroid (miles)
    
    schools = list(range(20))
    vans = [0, 1]
    levels = [1, 2, 3, 4]
    risks = [0, 1]  # 0=normal, 1=elevated
    
    # Parameters
    f_v = 500  # fixed cost per van
    g_v = 10   # variable cost per patient
    lam = 0.01
    pi_penalty = 5  # NO SHOW penalty
    w = {1: 1, 2: 2, 3: 3, 4: 4}
    alpha = {
        (4, 0): 1, (4, 1): 1,
        (3, 0): 0.7, (3, 1): 0.9,
        (2, 0): 0.4, (2, 1): 0.6,
        (1, 0): 0.3, (1, 1): 0.3,
    }
    M = 500
    
    mu_dist = 0.5  # weight for distance penalty (travel cost per mile)
    use_grb = use_gurobi if use_gurobi is not None else HAS_GUROBI
    if use_grb and HAS_GUROBI:
        res = _solve_gurobi(p, psi, patient_details, c_van, a_school, d_s,
                            schools, vans, levels, risks,
                            f_v, g_v, lam, pi_penalty, mu_dist, w, alpha, M)
        res["solver"] = "gurobi"
        return res
    res = _solve_pulp(p, psi, patient_details, c_van, a_school, d_s,
                     schools, vans, levels, risks,
                     f_v, g_v, lam, pi_penalty, mu_dist, w, alpha, M)
    res["solver"] = "pulp"
    return res


def _solve_gurobi(p, psi, patient_details, c_van, a_school, d_s,
                  schools, vans, levels, risks,
                  f_v, g_v, lam, pi_penalty, mu_dist, w, alpha, M):
    """Solve with Gurobi."""
    m = gp.Model("MCF_Schedule")
    m.setParam("OutputFlag", 0)
    
    x = {}
    for s in schools:
        for v in vans:
            for l in levels:
                for r in risks:
                    if p[s][(l, r)] > 0:
                        x[s, v, l, r] = m.addVar(lb=0, ub=p[s][(l, r)], vtype=GRB.INTEGER, name=f"x_{s}_{v}_{l}_{r}")
    y = {(s, v): m.addVar(vtype=GRB.BINARY, name=f"y_{s}_{v}") for s in schools for v in vans}
    z = {v: m.addVar(vtype=GRB.BINARY, name=f"z_{v}") for v in vans}
    m.update()
    
    obj_benefit = gp.quicksum(w[l] * x[s, v, l, r] for (s, v, l, r) in x)
    obj_cost = f_v * gp.quicksum(z[v] for v in vans)
    obj_var = g_v * gp.quicksum(x[k] for k in x)
    obj_noshow = pi_penalty * gp.quicksum(psi[s][(l, r)] * x[s, v, l, r] for (s, v, l, r) in x if psi[s][(l, r)] > 0)
    obj_distance = mu_dist * gp.quicksum(d_s.get(s, 0) * y[s, v] for s in schools for v in vans if s in d_s)
    m.setObjective(obj_benefit - lam * (obj_cost + obj_var + obj_noshow + obj_distance), GRB.MAXIMIZE)
    
    for v in vans:
        m.addConstr(gp.quicksum(x[k] for k in x if k[1] == v) <= c_van * z[v])
    for s in schools:
        m.addConstr(gp.quicksum(x[k] for k in x if k[0] == s) <= a_school)
    for s in schools:
        for l in levels:
            for r in risks:
                if p[s][(l, r)] > 0:
                    m.addConstr(gp.quicksum(x[k] for k in x if k[0] == s and k[2] == l and k[3] == r) <= p[s][(l, r)])
    for s in schools:
        for l in levels:
            for r in risks:
                if p[s][(l, r)] > 0 and alpha[(l, r)] == 1:
                    m.addConstr(gp.quicksum(x[k] for k in x if k[0] == s and k[2] == l and k[3] == r) >= p[s][(l, r)])
    for v in vans:
        m.addConstr(gp.quicksum(x[k] for k in x if k[1] == v) <= M * z[v])
    for s in schools:
        for v in vans:
            m.addConstr(gp.quicksum(x[k] for k in x if k[0] == s and k[1] == v) <= M * y[s, v])
    
    m.optimize()
    
    def get_val(v):
        return v.X if hasattr(v, 'X') else 0
    status = m.Status
    obj_val = m.objVal if status == GRB.OPTIMAL else 0
    
    schedule = {v: [] for v in vans}
    for (s, v, l, r), var in x.items():
        val = get_val(var)
        if val and val > 0:
            schedule[v].append({"school": s, "l": l, "r": r, "count": int(val)})
    
    assigned = {v: [] for v in vans}
    for (s, v, l, r), var in x.items():
        cnt = int(get_val(var) or 0)
        if cnt <= 0:
            continue
        pool = [d for d in patient_details if d["school"] == s and d["severity"] == l and d["elevated"] == r]
        pool.sort(key=lambda d: (-d["elevated"], -d["severity"]))
        for d in pool[:cnt]:
            assigned[v].append({"pid": d["pid"], "school": s, "severity": l, "elevated": r})
    
    return {
        "status": 1 if status == GRB.OPTIMAL else status,
        "objective": obj_val,
        "schedule": schedule,
        "assigned_patients": assigned,
        "z": {v: z[v].X for v in vans},
        "y": {(s, v): y[s, v].X for s in schools for v in vans},
        "p": p, "psi": psi, "patient_details": patient_details,
        "c_van": c_van, "a_school": a_school,
    }


def _solve_pulp(p, psi, patient_details, c_van, a_school, d_s,
                schools, vans, levels, risks,
                f_v, g_v, lam, pi_penalty, mu_dist, w, alpha, M):
    """Solve with PuLP/CBC (fallback when Gurobi not available)."""
    prob = LpProblem("MCF_Schedule", LpMaximize)
    
    x = {}
    for s in schools:
        for v in vans:
            for l in levels:
                for r in risks:
                    if p[s][(l, r)] > 0:
                        x[s, v, l, r] = LpVariable(f"x_{s}_{v}_{l}_{r}", lowBound=0, upBound=p[s][(l, r)], cat=LpInteger)
    y = {(s, v): LpVariable(f"y_{s}_{v}", cat="Binary") for s in schools for v in vans}
    z = {v: LpVariable(f"z_{v}", cat="Binary") for v in vans}
    
    obj_benefit = lpSum(w[l] * x[k] for k in x)
    obj_cost = lpSum(f_v * z[v] for v in vans)
    obj_var = lpSum(g_v * x[k] for k in x)
    obj_noshow = lpSum(pi_penalty * psi[s][(l, r)] * x[s, v, l, r] for (s, v, l, r) in x if psi[s][(l, r)] > 0)
    obj_distance = lpSum(mu_dist * d_s.get(s, 0) * y[s, v] for s in schools for v in vans if s in d_s)
    prob += obj_benefit - lam * (obj_cost + obj_var + obj_noshow + obj_distance)
    
    for v in vans:
        prob += lpSum(x[k] for k in x if k[1] == v) <= c_van * z[v]
    for s in schools:
        prob += lpSum(x[k] for k in x if k[0] == s) <= a_school
    for s in schools:
        for l in levels:
            for r in risks:
                if p[s][(l, r)] > 0:
                    prob += lpSum(x[k] for k in x if k[0] == s and k[2] == l and k[3] == r) <= p[s][(l, r)]
    for s in schools:
        for l in levels:
            for r in risks:
                if p[s][(l, r)] > 0 and alpha[(l, r)] == 1:
                    prob += lpSum(x[k] for k in x if k[0] == s and k[2] == l and k[3] == r) >= p[s][(l, r)]
    for v in vans:
        prob += lpSum(x[k] for k in x if k[1] == v) <= M * z[v]
    for s in schools:
        for v in vans:
            prob += lpSum(x[k] for k in x if k[0] == s and k[1] == v) <= M * y[s, v]
    
    prob.solve(PULP_CBC_CMD(msg=0))
    
    schedule = {v: [] for v in vans}
    for (s, v, l, r), var in x.items():
        val = var.varValue
        if val and val > 0:
            schedule[v].append({"school": s, "l": l, "r": r, "count": int(val)})
    
    assigned = {v: [] for v in vans}
    for (s, v, l, r), var in x.items():
        cnt = int(var.varValue or 0)
        if cnt <= 0:
            continue
        pool = [d for d in patient_details if d["school"] == s and d["severity"] == l and d["elevated"] == r]
        pool.sort(key=lambda d: (-d["elevated"], -d["severity"]))
        for d in pool[:cnt]:
            assigned[v].append({"pid": d["pid"], "school": s, "severity": l, "elevated": r})
    
    return {
        "status": prob.status,
        "objective": prob.objective.value(),
        "schedule": schedule,
        "assigned_patients": assigned,
        "z": {v: z[v].varValue for v in vans},
        "y": {(s, v): y[s, v].varValue for s in schools for v in vans},
        "p": p, "psi": psi, "patient_details": patient_details,
        "c_van": c_van, "a_school": a_school,
    }


def format_schedule(result, show_patient_ids=False):
    sev_names = {1: "MildIntermittent", 2: "MildPersistent", 3: "ModeratePersistent", 4: "SeverePersistent"}
    risk_names = {0: "normal", 1: "elevated"}
    
    lines = []
    lines.append("=" * 60)
    lines.append("MCF MONTHLY VAN SCHEDULE")
    lines.append("=" * 60)
    lines.append(f"Status: {'Optimal' if result['status'] == 1 else result['status']}")
    lines.append(f"Objective value: {result['objective']:.2f}")
    lines.append("")
    
    for v in [0, 1]:
        lines.append(f"--- Van {v} ---")
        lines.append(f"  Van used: {'Yes' if result['z'][v] else 'No'}")
        items = result["schedule"].get(v, [])
        if not items:
            lines.append("  No assignments.")
        else:
            by_school = {}
            for item in items:
                s = item["school"]
                if s not in by_school:
                    by_school[s] = []
                by_school[s].append(f"    Sev {sev_names[item['l']]} ({risk_names[item['r']]}): {item['count']} patients")
            for s in sorted(by_school.keys()):
                lines.append(f"  School {s}:")
                for line in by_school[s]:
                    lines.append(line)
                total = sum(x["count"] for x in items if x["school"] == s)
                lines.append(f"    Total: {total} patients")
        lines.append("")
    
    lines.append("--- Summary ---")
    total_0 = sum(x["count"] for x in result["schedule"].get(0, []))
    total_1 = sum(x["count"] for x in result["schedule"].get(1, []))
    lines.append(f"Van 0 total patients: {total_0}")
    lines.append(f"Van 1 total patients: {total_1}")
    lines.append(f"Total served: {total_0 + total_1}")
    
    return "\n".join(lines)

def run_eda():
    """Basic EDA for model parameters."""
    xl, loc, patient_to_school, patient_severity = load_data()
    p, psi, _ = build_patient_pool(xl, patient_to_school, patient_severity)
    c_van, a_school = infer_capacities(xl)
    
    print("=== EDA: Key Parameters ===\n")
    print("Patients per school (total):")
    for s in range(20):
        n = sum(p[s].values())
        if n > 0:
            print(f"  School {s}: {n}")
    print("\nPatients by severity (aggregate):")
    sev_totals = {1: 0, 2: 0, 3: 0, 4: 0}
    for s in range(20):
        for (l, r), cnt in p[s].items():
            sev_totals[l] += cnt
    sev_names = {1: "MildIntermittent", 2: "MildPersistent", 3: "ModeratePersistent", 4: "SeverePersistent"}
    for l in [4, 3, 2, 1]:
        print(f"  {sev_names[l]}: {sev_totals[l]}")
    print(f"\nInferred van capacity (per month): {c_van}")
    print(f"School capacity (assumed): {a_school}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "eda":
        run_eda()
    else:
        use_grb = "--gurobi" in sys.argv or "-g" in sys.argv
        use_pulp = "--pulp" in sys.argv or "-p" in sys.argv
        if use_grb:
            result = solve_mcf(use_gurobi=True)
        elif use_pulp:
            result = solve_mcf(use_gurobi=False)
        else:
            result = solve_mcf()
        solver = result.get("solver", "pulp")
        if use_grb and solver == "pulp":
            print("Note: Gurobi requested but not available, using PuLP/CBC.\n")
        print(f"(Using {solver.upper()})\n")
        print(format_schedule(result))
