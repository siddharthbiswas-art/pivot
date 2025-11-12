# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from pathlib import Path
import base64

st.set_page_config(layout="wide", page_title="Cohort M0-M11 Dashboard")

# ---------- Helpers ----------
def safe_to_datetime(s):
    return pd.to_datetime(s, infer_datetime_format=True, errors='coerce')

def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def compute_months(df, order_col, first_col):
    df['_order_dt'] = safe_to_datetime(df[order_col])
    df['_first_dt'] = safe_to_datetime(df[first_col])
    df['_order_month'] = df['_order_dt'].dt.to_period('M').dt.to_timestamp()
    df['_first_month'] = df['_first_dt'].dt.to_period('M').dt.to_timestamp()

    def month_offset(row):
        if pd.isna(row['_order_month']) or pd.isna(row['_first_month']):
            return np.nan
        return (row['_order_month'].year - row['_first_month'].year) * 12 + (row['_order_month'].month - row['_first_month'].month)

    df['month_number'] = df.apply(month_offset, axis=1).astype('Int64')
    return df

def pivot_metric(agg_df, metric, group_cols):
    """
    Create pivot table for metric grouped by `group_cols` and month_number.
    Ensures M0..M11 columns exist and returns columns ordered as:
    group_cols + M0..M11
    """
    pivot = agg_df.pivot_table(
        index=group_cols,
        columns='month_number',
        values=metric,
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    # rename numeric month columns to M0..M11
    rename = {}
    for c in pivot.columns:
        # pivot.columns may contain ints (0,1,..) or strings like '0'
        try:
            if isinstance(c, (int,)) or (isinstance(c, str) and c.isdigit()):
                rename[c] = f"M{int(c)}"
        except Exception:
            continue
    pivot = pivot.rename(columns=rename)

    # ensure all M0..M11 columns exist (use DataFrame column check, not dict.setdefault)
    for i in range(12):
        col = f"M{i}"
        if col not in pivot.columns:
            pivot[col] = 0

    # ensure ordering of columns: group_cols then M0..M11
    ordered_cols = list(group_cols) + [f"M{i}" for i in range(12)]
    # if some grouping columns are missing (unlikely), add them as UNKNOWN
    for gc in group_cols:
        if gc not in pivot.columns:
            pivot[gc] = "UNKNOWN"

    return pivot[ordered_cols]


def bytes_to_download(df, fmt='csv'):
    if fmt == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    else:
        # excel
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine='xlsxwriter')
        buffer.seek(0)
        return buffer.read()

# ---------- UI: Upload / Load ----------
st.title("Cohort M0 → M11 Dashboard (Streamlit)")

st.markdown("Upload a raw orders CSV (recommended) or provide precomputed pivot files named `pivot_revenue_M0_M11.csv`, `pivot_orders_M0_M11.csv`, `pivot_units_M0_M11.csv` in the working dir.")

upload = st.file_uploader("Upload CSV (raw)", type=['csv'], accept_multiple_files=False)
use_local_pivots = st.checkbox("Use existing pivot files in app directory (skip upload)", value=False)

# If user uploaded CSV, read it
df = None
if upload is not None and not use_local_pivots:
    df = pd.read_csv(upload, low_memory=False)
    st.success(f"Loaded uploaded CSV: {getattr(upload,'name','uploaded')}, rows: {len(df)}")

# If using local pivots, try to load them
DATA_DIR = Path(".")
if use_local_pivots and df is None:
    try:
        rev = pd.read_csv(DATA_DIR / "pivot_revenue_M0_M11.csv")
        ords = pd.read_csv(DATA_DIR / "pivot_orders_M0_M11.csv")
        units = pd.read_csv(DATA_DIR / "pivot_units_M0_M11.csv")
        st.success("Loaded local pivot files.")
    except Exception as e:
        st.error("Could not find pivot files in repo root. Upload raw CSV or add pivot files.")
        st.stop()

# ---------- If raw CSV, detect columns and preprocess ----------
if df is not None:
    # candidate lists (edit here if your column names differ)
    possible_order_date_cols = ['order_date','order_timestamp','order_created','order_date_as_dd_mm_yyyy_hh_mm_ss','order_date_pk_tz']
    possible_first_order_cols = ['first_order_date','first_order_month','first_order_date_as_string']
    possible_gmv_cols = ['gmv','revenue','order_value','total_price','selling_price','price','amount','order_amount']
    possible_qty_cols = ['quantity','qty','units','sku_qty','item_qty']
    possible_order_id_cols = ['order_id','sale_order_code','unique_order_id','display_order_code']

    order_col = pick_column(df, possible_order_date_cols)
    first_col = pick_column(df, possible_first_order_cols)
    gmv_col = pick_column(df, possible_gmv_cols)
    qty_col = pick_column(df, possible_qty_cols)
    orderid_col = pick_column(df, possible_order_id_cols)

    if order_col is None or first_col is None:
        st.error("Could not auto-detect `order_date` and/or `first_order_date` columns. Please ensure your CSV has one of these names (or edit the candidate lists in the app).")
        st.stop()

    st.markdown(f"**Detected** order column: `{order_col}`, first order column: `{first_col}`, revenue column: `{gmv_col}`, qty column: `{qty_col}`, order id: `{orderid_col}`")

    df = compute_months(df, order_col, first_col)

    # Revenue
    if gmv_col:
        # remove commas, coerce
        df['__revenue'] = pd.to_numeric(df[gmv_col].astype(str).str.replace(',',''), errors='coerce').fillna(0.0)
    else:
        df['__revenue'] = 0.0

    # Units
    if qty_col:
        df['__units'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
    else:
        df['__units'] = 1

    # Orders
    if orderid_col:
        df['__order_id'] = df[orderid_col].astype(str)
    else:
        df['__order_id'] = df.index.astype(str)

    # grouping columns detection
    group_cols_detect = {
        "cohort_col": pick_column(df, ['cohort','cohort_name']) or 'cohort',
        "business_line_col": pick_column(df, ['business_line','biz_line','bizline']) or 'business_line',
        "biz_category_col": pick_column(df, ['biz_category','category','article_type','biz_category']) or 'biz_category',
        "active_col": pick_column(df, ['active_vs_expired','active_status','membership_status']) or 'active_vs_expired'
    }
    for k,v in group_cols_detect.items():
        if v not in df.columns:
            df[v] = 'UNKNOWN'

    # optional date filter (order month)
    min_order = df['_order_month'].min()
    max_order = df['_order_month'].max()
    st.sidebar.markdown("### Date range filter (order month)")
    date_from = st.sidebar.date_input("From", value=min_order.date() if pd.notna(min_order) else None)
    date_to = st.sidebar.date_input("To", value=max_order.date() if pd.notna(max_order) else None)

    # apply date filter
    if date_from and date_to:
        df = df[(df['_order_month'] >= pd.to_datetime(date_from)) & (df['_order_month'] <= pd.to_datetime(date_to))]

    # filter month_number 0..11 and valid first month
    df = df[df['_first_month'].notna() & df['month_number'].notna()]
    df['month_number'] = df['month_number'].astype(int)
    df = df[df['month_number'].between(0,11)]

    # aggregate
    agg = df.groupby([group_cols_detect['cohort_col'], group_cols_detect['business_line_col'], group_cols_detect['biz_category_col'], group_cols_detect['active_col'], 'month_number']).agg(
        revenue = ('__revenue','sum'),
        orders = ('__order_id', lambda x: x.nunique()),
        units  = ('__units','sum')
    ).reset_index()

    # produce pivots
    group_cols_list = [group_cols_detect['cohort_col'], group_cols_detect['business_line_col'], group_cols_detect['biz_category_col'], group_cols_detect['active_col']]
    pivot_revenue = pivot_metric(agg, 'revenue', group_cols_list)
    pivot_orders  = pivot_metric(agg, 'orders', group_cols_list)
    pivot_units   = pivot_metric(agg, 'units', group_cols_list)

    # Save to disk for convenience (so you can commit these if you want)
    pivot_revenue.to_csv("pivot_revenue_M0_M11.csv", index=False)
    pivot_orders.to_csv("pivot_orders_M0_M11.csv", index=False)
    pivot_units.to_csv("pivot_units_M0_M11.csv", index=False)

else:
    # using local pivots (loaded earlier)
    # rev, ords, units available
    pivot_revenue = rev
    pivot_orders = ords
    pivot_units = units
    # detect group column names mapping (normalize names)
    # we assume columns: cohort, business_line, biz_category, active_vs_expired
    # if names differ, user should prepare pivot files accordingly

# ---------- Filters & Selection ----------
# Determine available values
def safe_unique(df, col): return ['All'] + sorted(df[col].dropna().unique().tolist()) if col in df.columns else ['All']

cohort_col = 'cohort'
bizline_col = 'business_line'
bizcat_col = 'biz_category'
active_col = 'active_vs_expired'

sel_cohort = st.sidebar.selectbox("Cohort", safe_unique(pivot_revenue, cohort_col))
sel_bizline = st.sidebar.selectbox("Business Line", safe_unique(pivot_revenue, bizline_col))
sel_active = st.sidebar.selectbox("Active vs Expired", safe_unique(pivot_revenue, active_col))
sel_metric = st.sidebar.selectbox("Metric", ["Revenue","Orders","Units","Retention %"], index=0)

# allow choosing row dimension and column metric for pivoting
row_dim = st.sidebar.selectbox("Row dimension", [bizcat_col, bizline_col, cohort_col, active_col], index=0)
display_mode = st.sidebar.selectbox("Display mode", ["Absolute values", "Percent of M0"], index=0)

# Apply filters on the base pivot chosen
if sel_metric == "Revenue" or sel_metric == "Retention %":
    base_df = pivot_revenue.copy()
elif sel_metric == "Orders":
    base_df = pivot_orders.copy()
else:
    base_df = pivot_units.copy()

df_display = base_df.copy()
if sel_cohort != 'All':
    df_display = df_display[df_display[cohort_col] == sel_cohort]
if sel_bizline != 'All':
    df_display = df_display[df_display[bizline_col] == sel_bizline]
if sel_active != 'All':
    df_display = df_display[df_display[active_col] == sel_active]

# Optionally choose biz categories to display
cats = sorted(df_display[bizcat_col].unique().tolist())
sel_cats = st.sidebar.multiselect("Biz categories (rows) - empty = all", options=cats, default=cats[:10])
if sel_cats:
    df_display = df_display[df_display[bizcat_col].isin(sel_cats)]

# ---------- Compute retention % if requested ----------
m_cols = [f"M{i}" for i in range(12)]
for col in m_cols:
    if col not in df_display.columns:
        df_display[col] = 0

if sel_metric == "Retention %":
    # compute percent of M0 per row
    def pct_row(r):
        m0 = r.get('M0', 0) or 0
        out = {}
        for i in range(12):
            out[f"M{i}"] = (r.get(f"M{i}",0) / m0 * 100) if m0 != 0 else np.nan
        return pd.Series(out)
    pct_df = df_display.apply(pct_row, axis=1)
    df_vis = pd.concat([df_display[[row_dim]], pct_df.reset_index(drop=True)], axis=1)
    heat_values = pct_df.fillna(0).values
    heat_index = df_display[row_dim].tolist()
    color_label = "% of M0"
else:
    # absolute values
    df_vis = df_display[[row_dim] + m_cols].groupby(row_dim)[m_cols].sum().reset_index()
    heat_values = df_vis[m_cols].values
    heat_index = df_vis[row_dim].tolist()
    color_label = sel_metric

# ---------- Heatmap ----------
st.subheader(f"{sel_metric} heatmap by {row_dim} (M0..M11)")
fig = px.imshow(heat_values,
                labels=dict(x="Month (M0..M11)", y=row_dim, color=color_label),
                x=m_cols,
                y=heat_index,
                aspect="auto")
st.plotly_chart(fig, use_container_width=True)

# ---------- Table and download ----------
st.markdown("### Table (filtered)")
st.dataframe(df_vis.style.format({c: "{:,.2f}" for c in m_cols}))

# Downloads
st.markdown("#### Export filtered table")
col1, col2 = st.columns(2)
with col1:
    fmt = st.selectbox("Format", ["csv","xlsx"])
with col2:
    filename = st.text_input("Filename (without extension)", value="cohort_pivot_filtered")

if st.button("Download"):
    b = bytes_to_download(df_vis, fmt=fmt)
    st.download_button(label="Download file", data=b, file_name=f"{filename}.{fmt}", mime=("text/csv" if fmt=='csv' else "application/vnd.ms-excel"))

st.markdown("**Notes:** M0 is user's first order month (cohort). Percent mode computes each row's months as % of that row's M0. Blank or zero M0 will appear as NaN in percent mode.")
