# streamlit_app.py
# Cohort M0-M11 Dashboard - Full version with per-biz-category small multiples & trend charts
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from pathlib import Path
from math import ceil

st.set_page_config(layout="wide", page_title="Cohort M0-M11 Dashboard (Trends by Biz Category)")

# ---------------- Helpers ----------------
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
    pivot = agg_df.pivot_table(index=group_cols, columns='month_number', values=metric, aggfunc='sum', fill_value=0).reset_index()
    rename = {}
    for c in pivot.columns:
        try:
            if isinstance(c, int) or (isinstance(c, str) and c.isdigit()):
                rename[c] = f"M{int(c)}"
        except Exception:
            continue
    pivot = pivot.rename(columns=rename)
    for i in range(12):
        col = f"M{i}"
        if col not in pivot.columns:
            pivot[col] = 0
    ordered_cols = list(group_cols) + [f"M{i}" for i in range(12)]
    for gc in group_cols:
        if gc not in pivot.columns:
            pivot[gc] = "UNKNOWN"
    return pivot[ordered_cols]

def bytes_to_download(df, fmt='csv'):
    if fmt == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    else:
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine='xlsxwriter')
        buffer.seek(0)
        return buffer.read()

def ensure_mcols(df):
    for i in range(12):
        c = f"M{i}"
        if c not in df.columns:
            df[c] = 0
    return df

# ---------------- UI: upload / local pivot ----------------
st.title("Cohort M0 → M11 Dashboard — Trends & Small Multiples by Biz Category")

upload = st.file_uploader("Upload raw orders CSV (recommended)", type=['csv'])
use_local_pivots = st.checkbox("Use existing pivot files in app directory (skip upload)", value=False)

df = None
if upload is not None and not use_local_pivots:
    df = pd.read_csv(upload, low_memory=False)
    st.success(f"Loaded uploaded CSV: {getattr(upload,'name','uploaded')}, rows: {len(df)}")

if use_local_pivots and df is None:
    # attempt to load pivot files
    try:
        pivot_revenue = pd.read_csv("pivot_revenue_M0_M11.csv")
        pivot_orders = pd.read_csv("pivot_orders_M0_M11.csv")
        pivot_units = pd.read_csv("pivot_units_M0_M11.csv")
        pivot_users = pd.read_csv("pivot_users_M0_M11.csv") if Path("pivot_users_M0_M11.csv").exists() else pivot_orders.copy()
        st.success("Loaded pivot files from app directory.")
    except Exception as e:
        st.error("Could not load pivot files from app directory. Upload a raw CSV or add pivot files.")
        st.stop()

# ---------------- Preprocess raw CSV ----------------
if df is not None:
    # candidate column lists
    possible_order_date_cols = ['order_date','order_timestamp','order_created','order_date_as_dd_mm_yyyy_hh_mm_ss','order_date_pk_tz']
    possible_first_order_cols = ['first_order_date','first_order_month','first_order_date_as_string']
    possible_gmv_cols = ['gmv','revenue','order_value','total_price','selling_price','price','amount','order_amount']
    possible_qty_cols = ['quantity','qty','units','sku_qty','item_qty']
    possible_order_id_cols = ['order_id','sale_order_code','unique_order_id','display_order_code']
    possible_user_id_cols = ['user_id','customer_id','user']

    order_col = pick_column(df, possible_order_date_cols)
    first_col = pick_column(df, possible_first_order_cols)
    gmv_col = pick_column(df, possible_gmv_cols)
    qty_col = pick_column(df, possible_qty_cols)
    orderid_col = pick_column(df, possible_order_id_cols)
    user_col = pick_column(df, possible_user_id_cols)

    if order_col is None or first_col is None:
        st.error("Could not detect order_date and/or first_order_date columns. Edit the candidate lists in the app if your column names differ.")
        st.stop()

    st.markdown(f"Detected -> order: `{order_col}`, first order: `{first_col}`, revenue: `{gmv_col}`, qty: `{qty_col}`, order_id: `{orderid_col}`, user_id: `{user_col}`")

    # months & offsets
    df = compute_months(df, order_col, first_col)

    # revenue
    if gmv_col:
        df['__revenue'] = pd.to_numeric(df[gmv_col].astype(str).str.replace(',',''), errors='coerce').fillna(0.0)
    else:
        df['__revenue'] = 0.0

    # units
    if qty_col:
        df['__units'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(0)
    else:
        df['__units'] = 1

    # orders
    if orderid_col:
        df['__order_id'] = df[orderid_col].astype(str)
    else:
        df['__order_id'] = df.index.astype(str)

    # group columns expected in your data
    # use these names if present; otherwise create fallbacks
    for col in ['business_line','biz_category','is_fs_member','new_vs_repeat_orders','cohort','active_vs_expired']:
        if col not in df.columns:
            # create fallbacks or defaults
            if col == 'new_vs_repeat_orders':
                df[col] = np.nan
            elif col == 'is_fs_member':
                df[col] = np.nan
            else:
                df[col] = 'UNKNOWN'

    # allow user to filter order month range and cohort month (first_order_month)
    st.sidebar.markdown("### Filters")
    min_order = df['_order_month'].min()
    max_order = df['_order_month'].max()
    date_from = st.sidebar.date_input("Order month from", value=min_order.date() if pd.notna(min_order) else None)
    date_to = st.sidebar.date_input("Order month to", value=max_order.date() if pd.notna(max_order) else None)
    if date_from and date_to:
        df = df[(df['_order_month'] >= pd.to_datetime(date_from)) & (df['_order_month'] <= pd.to_datetime(date_to))]

    # restrict to month_number 0..11 and valid first month
    df = df[df['_first_month'].notna() & df['month_number'].notna()]
    df['month_number'] = df['month_number'].astype(int)
    df = df[df['month_number'].between(0,11)]

    # compute unique users per group if user_id present
    if user_col:
        users_agg = df.groupby(['cohort','business_line','biz_category','active_vs_expired','month_number'])[user_col].nunique().reset_index(name='users')
    else:
        users_agg = df.groupby(['cohort','business_line','biz_category','active_vs_expired','month_number'])['__order_id'].nunique().reset_index(name='users')

    # aggregate revenue/orders/units
    agg = df.groupby(['cohort','business_line','biz_category','active_vs_expired','month_number']).agg(
        revenue = ('__revenue','sum'),
        orders = ('__order_id', lambda x: x.nunique()),
        units  = ('__units','sum')
    ).reset_index()

    # merge users
    agg = agg.merge(users_agg, on=['cohort','business_line','biz_category','active_vs_expired','month_number'], how='left')

    # produce pivots
    group_cols_list = ['cohort','business_line','biz_category','active_vs_expired']
    pivot_revenue = pivot_metric(agg, 'revenue', group_cols_list)
    pivot_orders = pivot_metric(agg, 'orders', group_cols_list)
    pivot_units = pivot_metric(agg, 'units', group_cols_list)
    pivot_users = pivot_metric(agg, 'users', group_cols_list)

    # Save optional pivot files for convenience
    pivot_revenue.to_csv("pivot_revenue_M0_M11.csv", index=False)
    pivot_orders.to_csv("pivot_orders_M0_M11.csv", index=False)
    pivot_units.to_csv("pivot_units_M0_M11.csv", index=False)
    pivot_users.to_csv("pivot_users_M0_M11.csv", index=False)

else:
    # if using local pivots, ensure they exist
    try:
        pivot_revenue
    except NameError:
        st.error("No data loaded. Upload a CSV or enable 'Use existing pivot files'.")
        st.stop()

# --------------- Common filters/controls ---------------
# Cohort selection (first order month)
cohort_values = sorted(pivot_users['cohort'].dropna().unique().tolist())
cohort_vals_with_all = ["All"] + cohort_values
sel_cohort_month = st.sidebar.selectbox("Cohort (first_order_month)", cohort_vals_with_all, index=0)

# Business line and active filters
bizline_values = ["All"] + sorted(pivot_revenue['business_line'].dropna().unique().tolist())
sel_bizline_filter = st.sidebar.selectbox("Business Line (filter)", bizline_values, index=0)
sel_active_filter = st.sidebar.selectbox("Active vs Expired (filter)", ["All"] + sorted(pivot_revenue['active_vs_expired'].dropna().unique().tolist()), index=0)

# small multiples toggle and selection of biz categories
small_multiples = st.sidebar.checkbox("Small multiples per Biz Category (one chart per biz_category)", value=True)
biz_categories_available = sorted(pivot_revenue['biz_category'].dropna().unique().tolist())
sel_biz_cats = st.sidebar.multiselect("Biz Categories to include (small multiples)", options=biz_categories_available, default=biz_categories_available[:6])

# Trend grouping selection (what each line represents)
trend_grouping = st.sidebar.selectbox("Trend grouping (each line on chart)", ["business_line","new_vs_repeat","is_fs_member","cohort"], index=0)

# For new_vs_repeat, ensure we have a source: prefer new_vs_repeat_orders column in raw df if available
# If raw df exists, check column
if 'df' in globals() and df is not None:
    nvr_col_raw = pick_column(df, ['new_vs_repeat_orders','new_vs_repeat','new_vs_repeat_fs'])
else:
    nvr_col_raw = None

# --------------- Prepare unified trend dataframe ---------------
# We'll use the raw df if available to compute new_vs_repeat classification and is_fs_member,
# otherwise derive from the pivot tables (best effort)
if 'df' in globals() and df is not None:
    df_trend = df.copy()
    # new vs repeat classification
    if nvr_col_raw:
        df_trend['__new_repeat'] = df_trend[nvr_col_raw].astype(str).str.upper().fillna('UNKNOWN')
    else:
        df_trend['__new_repeat'] = df_trend['month_number'].apply(lambda x: 'NEW' if x == 0 else 'REPEAT')

    # is_fs_member: try to coerce to string for grouping
    if 'is_fs_member' in df_trend.columns:
        df_trend['__is_fs_member'] = df_trend['is_fs_member'].fillna('UNKNOWN').astype(str)
    else:
        df_trend['__is_fs_member'] = 'UNKNOWN'

    # cohort already exists as first_order_month string values in 'cohort' column
    df_trend['cohort'] = df_trend['cohort'].fillna('UNKNOWN').astype(str)
    df_trend['business_line'] = df_trend['business_line'].fillna('UNKNOWN').astype(str)
    df_trend['biz_category'] = df_trend['biz_category'].fillna('UNKNOWN').astype(str)

    # aggregate to month-level per groupings we need
    # columns: biz_category, business_line, __new_repeat, __is_fs_member, cohort, month_number, revenue, orders
    trend_agg = df_trend.groupby(['biz_category','business_line','__new_repeat','__is_fs_member','cohort','month_number']).agg(
        revenue = ('__revenue','sum'),
        orders = ('__order_id', lambda x: x.nunique())
    ).reset_index().rename(columns={'__new_repeat':'new_vs_repeat','__is_fs_member':'is_fs_member'})

else:
    # Build trend_agg from pivot tables as fallback
    # iterate pivot_revenue rows and melt M0..M11 into month_number rows
    def melt_pivot(pivot_df, value_col):
        rows = []
        for _, r in pivot_df.iterrows():
            base = {'cohort': r['cohort'], 'business_line': r['business_line'], 'biz_category': r['biz_category'], 'active_vs_expired': r['active_vs_expired']}
            for i in range(12):
                rows.append({
                    **base,
                    'month_number': i,
                    value_col: r.get(f"M{i}", 0)
                })
        return pd.DataFrame(rows)
    rev_melt = melt_pivot(pivot_revenue, 'revenue')
    ord_melt = melt_pivot(pivot_orders, 'orders')
    trend_agg = rev_melt.merge(ord_melt, on=['cohort','business_line','biz_category','active_vs_expired','month_number'], how='outer').fillna(0)
    # fallback columns
    trend_agg['new_vs_repeat'] = trend_agg['month_number'].apply(lambda x: 'NEW' if x==0 else 'REPEAT')
    trend_agg['is_fs_member'] = 'UNKNOWN'
    trend_agg['cohort'] = trend_agg['cohort'].astype(str)

# --------------- Apply global filters to trend_agg ---------------
trend_df = trend_agg.copy()
if sel_bizline_filter != 'All':
    trend_df = trend_df[trend_df['business_line'] == sel_bizline_filter]
if sel_active_filter != 'All' and 'active_vs_expired' in trend_df.columns:
    trend_df = trend_df[trend_df['active_vs_expired'] == sel_active_filter]
if sel_cohort_month != 'All':
    trend_df = trend_df[trend_df['cohort'] == sel_cohort_month]

# --------------- Trend charts (aggregate or small multiples) ---------------
st.header("Trend charts (Revenue & Orders) — M0 .. M11")

# Helper: build a plotting frame for the requested grouping and measure
def build_plot_df(trend_df, group_col, measure):
    # group_col: one of 'business_line','new_vs_repeat','is_fs_member','cohort'
    dfp = trend_df.copy()
    if group_col == 'new_vs_repeat':
        group_key = 'new_vs_repeat'
    elif group_col == 'is_fs_member':
        group_key = 'is_fs_member'
    else:
        group_key = group_col
    # choose measure column: 'revenue' or 'orders'
    plot_df = dfp.groupby([group_key,'biz_category','month_number'])[measure].sum().reset_index()
    # pivot so month_number across columns if needed; but for plotting tidy format is fine
    return plot_df, group_key

# choose grouping and measure for the main trend chart
main_trend_group = st.sidebar.selectbox("Main trend grouping (for overview charts)", ["business_line","new_vs_repeat","is_fs_member","cohort"], index=0)
main_measure = st.sidebar.selectbox("Main trend measure", ["revenue","orders"], index=0)

plot_df, plot_group_key = build_plot_df(trend_df, main_trend_group, main_measure)

# If small multiples enabled -> one chart per biz_category
if small_multiples:
    st.subheader(f"Small multiples by Biz Category — lines = {plot_group_key}, measure = {main_measure.upper()}")
    # limit biz categories to selection
    bizs = sel_biz_cats if sel_biz_cats else biz_categories_available
    # layout: ncols = 2 or 3 depending on number
    n = len(bizs)
    ncols = 2 if n <= 6 else 3
    rows = ceil(n / ncols)
    idx = 0
    for r in range(rows):
        cols = st.columns(ncols)
        for c in range(ncols):
            if idx >= n: 
                cols[c].empty()
                idx += 1
                continue
            biz = bizs[idx]
            sub = plot_df[plot_df['biz_category'] == biz]
            if sub.empty:
                cols[c].write(f"No data for {biz}")
            else:
                # tidy form: x=month_number, y=measure, color=group_key
                fig = px.line(sub, x='month_number', y=main_measure, color=plot_group_key, markers=True, title=f"{biz} — {main_measure}")
                fig.update_xaxes(tickmode='array', tickvals=list(range(12)), ticktext=[f"M{i}" for i in range(12)])
                cols[c].plotly_chart(fig, use_container_width=True)
            idx += 1
else:
    # Aggregate across biz categories (or show interactive select)
    st.subheader(f"Overview trend — lines = {plot_group_key}, measure = {main_measure.upper()} (aggregated across biz categories)")
    # Optionally allow selecting a few biz categories to show separately by facet
    chosen_biz = st.multiselect("Limit to Biz Categories (leave empty = all)", options=biz_categories_available, default=biz_categories_available[:6])
    df_plot = plot_df.copy()
    if chosen_biz:
        df_plot = df_plot[df_plot['biz_category'].isin(chosen_biz)]
    # aggregate (group_key, month_number)
    df_plot_agg = df_plot.groupby([plot_group_key,'month_number'])[main_measure].sum().reset_index()
    if df_plot_agg.empty:
        st.info("No data to plot for this selection.")
    else:
        fig = px.line(df_plot_agg, x='month_number', y=main_measure, color=plot_group_key, markers=True, title=f"{main_measure.title()} trend")
        fig.update_xaxes(tickmode='array', tickvals=list(range(12)), ticktext=[f"M{i}" for i in range(12)])
        st.plotly_chart(fig, use_container_width=True)

# --------------- Additional: small overview panels for Business Line, New vs Repeat, is_fs_member, Cohort ---------------
st.header("Quick multi-dimension trends")

# Build utility to draw two small charts (revenue and orders) for a given dimension
def draw_pair_for_dimension(dim_key, label):
    st.subheader(label)
    df_dim = trend_df.copy()
    if dim_key == 'new_vs_repeat':
        color_col = 'new_vs_repeat'
    elif dim_key == 'is_fs_member':
        color_col = 'is_fs_member'
    else:
        color_col = dim_key

    # revenue
    rev_dim = df_dim.groupby([color_col,'month_number'])['revenue'].sum().reset_index()
    ord_dim = df_dim.groupby([color_col,'month_number'])['orders'].sum().reset_index()

    if rev_dim.empty:
        st.info(f"No data for {label}")
        return

    c1, c2 = st.columns(2)
    fig_r = px.line(rev_dim, x='month_number', y='revenue', color=color_col, markers=True, title=f"{label} — Revenue")
    fig_r.update_xaxes(tickmode='array', tickvals=list(range(12)), ticktext=[f"M{i}" for i in range(12)])
    c1.plotly_chart(fig_r, use_container_width=True)

    fig_o = px.line(ord_dim, x='month_number', y='orders', color=color_col, markers=True, title=f"{label} — Orders")
    fig_o.update_xaxes(tickmode='array', tickvals=list(range(12)), ticktext=[f"M{i}" for i in range(12)])
    c2.plotly_chart(fig_o, use_container_width=True)

# Draw for Business Line
draw_pair_for_dimension('business_line', "Business Line")

# Draw for New vs Repeat
draw_pair_for_dimension('new_vs_repeat', "New vs Repeat")

# Draw for is_fs_member
draw_pair_for_dimension('is_fs_member', "is_fs_member (membership)")

# Draw for Cohort (limited list to avoid huge plots)
cohort_sample = sorted(trend_df['cohort'].dropna().unique().tolist())[:10]
st.subheader("Cohort (top 10 shown)")
cohort_dim = trend_df[trend_df['cohort'].isin(cohort_sample)]
if cohort_dim.empty:
    st.info("No cohort data to show.")
else:
    rev_cohort = cohort_dim.groupby(['cohort','month_number'])['revenue'].sum().reset_index()
    ord_cohort = cohort_dim.groupby(['cohort','month_number'])['orders'].sum().reset_index()
    fig_rc = px.line(rev_cohort, x='month_number', y='revenue', color='cohort', markers=True, title="Cohort — Revenue (top 10)")
    fig_rc.update_xaxes(tickmode='array', tickvals=list(range(12)), ticktext=[f"M{i}" for i in range(12)])
    st.plotly_chart(fig_rc, use_container_width=True)
    fig_oc = px.line(ord_cohort, x='month_number', y='orders', color='cohort', markers=True, title="Cohort — Orders (top 10)")
    fig_oc.update_xaxes(tickmode='array', tickvals=list(range(12)), ticktext=[f"M{i}" for i in range(12)])
    st.plotly_chart(fig_oc, use_container_width=True)

# --------------- Download trimmed trend data ---------------
st.header("Download trend data (filtered)")
download_df = trend_df.copy()
st.markdown("Filtered trend data (columns: biz_category, business_line, new_vs_repeat, is_fs_member, cohort, month_number, revenue, orders)")
st.dataframe(download_df.head(200))
fmt = st.selectbox("Download format", ["csv","xlsx"])
if st.button("Download trend data"):
    b = bytes_to_download(download_df, fmt=fmt)
    st.download_button(label="Download", data=b, file_name=f"trend_data.{fmt}", mime=("text/csv" if fmt=='csv' else "application/vnd.ms-excel"))

# --------------- Footer / notes ---------------
st.markdown("""
**Notes**
- X-axis months are M0..M11 where M0 = user's first order month (cohort).
- Small multiples: create one chart per selected Biz Category. Each chart's lines are the selected trend grouping (Business Line / New vs Repeat / is_fs_member / Cohort).
- If you have a `user_id` column the app computes unique users; otherwise it uses unique orders as a proxy.
- If your dataset uses different column names, update the top candidate lists (order_date, first_order_date, gmv, etc.).
""")
