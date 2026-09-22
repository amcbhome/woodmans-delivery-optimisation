import math
import pulp
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Woodman's Delivery Optimisation", layout="wide")

NODES = ["D", "C1", "C2", "C3", "C4", "C5"]
CUSTOMERS = NODES[1:]
VEHICLES = ["W", "R"]
DEMAND = {"C1": 4, "C2": 6, "C3": 3, "C4": 8, "C5": 5}
CAPACITY = {"W": 40, "R": 10}
COST = {"W": 1.38, "R": 1.85}
COORDS = {"D": (0,0), "C1": (3,4), "C2": (6,2), "C3": (2,7), "C4": (8,6), "C5": (5,9)}
DIST = {(i,j): math.hypot(COORDS[i][0]-COORDS[j][0], COORDS[i][1]-COORDS[j][1])
        for i in NODES for j in NODES if i != j}
SPEED, SERVICE, WINDOW_CLOSE = 0.5, 5, 350
BIG_M = 1000

def solve(penalty, max_w, max_r):
    m = pulp.LpProblem("Woodmans_Delivery", pulp.LpMinimize)
    x = pulp.LpVariable.dicts("x", ((i,j,k) for i in NODES for j in NODES if i!=j for k in VEHICLES), cat="Binary")
    f = pulp.LpVariable.dicts("f", ((i,j,k) for i in NODES for j in NODES if i!=j for k in VEHICLES), lowBound=0)
    p = pulp.LpVariable.dicts("p", ((i,k) for i in CUSTOMERS for k in VEHICLES), lowBound=0, upBound=WINDOW_CLOSE)
    reject = pulp.LpVariable.dicts("reject", CUSTOMERS, cat="Binary")

    m += (pulp.lpSum(COST[k]*DIST[i,j]*x[i,j,k] for i in NODES for j in NODES if i!=j for k in VEHICLES)
          + pulp.lpSum(penalty*DEMAND[i]*reject[i] for i in CUSTOMERS))

    limits = {"W": max_w, "R": max_r}
    for k in VEHICLES:
        m += pulp.lpSum(x["D",j,k] for j in CUSTOMERS) <= limits[k]
        m += pulp.lpSum(x[i,"D",k] for i in CUSTOMERS) <= limits[k]

    for i in CUSTOMERS:
        incoming = pulp.lpSum(x[j,i,k] for j in NODES if j!=i for k in VEHICLES)
        outgoing = pulp.lpSum(x[i,j,k] for j in NODES if j!=i for k in VEHICLES)
        m += incoming <= 1
        m += outgoing <= 1
        m += incoming + reject[i] == 1
        for k in VEHICLES:
            m += pulp.lpSum(x[i,j,k] for j in NODES if j!=i) == pulp.lpSum(x[j,i,k] for j in NODES if j!=i)

    for k in VEHICLES:
        for i in NODES:
            for j in CUSTOMERS:
                if i != j:
                    m += f[i,j,k] >= DEMAND[j]*x[i,j,k]
        for i in CUSTOMERS:
            for j in NODES:
                if i != j:
                    m += f[i,j,k] <= CAPACITY[k]*x[i,j,k]

    for i in CUSTOMERS:
        for k in VEHICLES:
            visit = pulp.lpSum(x[i,j,k] for j in NODES if j!=i)
            m += p[i,k] <= WINDOW_CLOSE*visit

    for k in VEHICLES:
        for j in CUSTOMERS:
            m += p[j,k] >= DIST["D",j]/SPEED - BIG_M*(1-x["D",j,k])
        for i in CUSTOMERS:
            for j in CUSTOMERS:
                if i != j:
                    m += p[j,k] >= p[i,k] + SERVICE + DIST[i,j]/SPEED - BIG_M*(1-x[i,j,k])

    status = m.solve(pulp.PULP_CBC_CMD(msg=False))
    return m, x, reject, pulp.LpStatus[status]

st.title("Woodman's Grocery Delivery Optimisation")
st.caption("Reduced 5-customer HVRPTW model implemented with PuLP and Streamlit.")

with st.sidebar:
    st.header("Model controls")
    penalty = st.slider("Penalty per rejected carton", 0.0, 20.0, 5.0, 0.5)
    max_w = st.slider("Woodman's vehicles", 0, 2, 2)
    max_r = st.slider("Rideshare vehicles", 0, 2, 2)
    solve_button = st.button("Solve optimisation", type="primary")

if solve_button or "result" not in st.session_state:
    st.session_state.result = solve(penalty, max_w, max_r)

model, x, reject, status = st.session_state.result

if status != "Optimal":
    st.error(f"Solver status: {status}")
else:
    served, rejected_rows, routes = [], [], []
    for i in CUSTOMERS:
        if pulp.value(reject[i]) > 0.5:
            rejected_rows.append({"Customer": i, "Demand": DEMAND[i], "Status": "Rejected"})
        else:
            served.append({"Customer": i, "Demand": DEMAND[i], "Status": "Served"})

    for k in VEHICLES:
        for start in CUSTOMERS:
            if pulp.value(x["D",start,k]) > 0.5:
                route = ["D", start]
                current = start
                while True:
                    nxt = next((j for j in NODES if j != current and pulp.value(x[current,j,k]) > 0.5), None)
                    if nxt is None:
                        break
                    route.append(nxt)
                    if nxt == "D":
                        break
                    current = nxt
                routes.append({"Vehicle": "Woodman's" if k=="W" else "Rideshare", "Route": " → ".join(route)})

    total_demand = sum(DEMAND.values())
    served_demand = sum(r["Demand"] for r in served)
    c1,c2,c3 = st.columns(3)
    c1.metric("Objective cost", f"{pulp.value(model.objective):.2f}")
    c2.metric("Demand served", f"{served_demand} / {total_demand}")
    c3.metric("Customers rejected", len(rejected_rows))

    st.subheader("Optimised routes")
    st.dataframe(pd.DataFrame(routes), use_container_width=True, hide_index=True)

    st.subheader("Customer allocation")
    st.dataframe(pd.DataFrame(served + rejected_rows), use_container_width=True, hide_index=True)

st.divider()
st.markdown("**Reference:** [NEOS Guide — Woodman's Delivery](https://neos-guide.org/case-studies/tra/woodmans-delivery/)")
st.caption("Educational reduced instance. Distances are illustrative, not the original 30-customer NEOS data.")
