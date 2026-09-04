import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

st.set_page_config(layout="wide", page_title="Survival Churn App", initial_sidebar_state="expanded")

if "cox_ready" not in st.session_state:
    st.session_state["cox_ready"] = False
if "cox_ready_set" not in st.session_state:
    st.session_state["cox_ready_set"] = False
if "view_choice_global" not in st.session_state:
    st.session_state["view_choice_global"] = "KM"

SCRIPT_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(SCRIPT_DIR, "Telco_customer_churn.csv")

@st.cache_data
def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    return df

@st.cache_data
def compute_km(df, duration_col, event_col, group_col):
    kmf = KaplanMeierFitter()
    results = {}
    for grp in sorted(df[group_col].dropna().unique()):
        mask = df[group_col] == grp
        try:
            kmf.fit(df[mask][duration_col], df[mask][event_col], label=str(grp))
            surv_df = kmf.survival_function_.reset_index()
            surv_df.columns = ["timeline", "survival"]
            results[grp] = surv_df
        except Exception:
            results[grp] = None
    return results

@st.cache_data
def compute_logrank_pairs(df, duration_col, event_col, group_col):
    cats = sorted(df[group_col].dropna().unique())
    results = {}
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            cat1, cat2 = cats[i], cats[j]
            mask1 = df[group_col] == cat1
            mask2 = df[group_col] == cat2
            res = logrank_test(df[mask1][duration_col], df[mask2][duration_col],
                               event_observed_A=df[mask1][event_col], event_observed_B=df[mask2][event_col])
            p = res.p_value
            hr = hr_lower = hr_upper = None
            try:
                subset = df[df[group_col].isin([cat1, cat2])][[duration_col, event_col, group_col]].dropna()
                subset = subset.copy()
                subset['__grp__'] = (subset[group_col] == cat2).astype(int)
                cph_tmp = CoxPHFitter()
                cph_tmp.fit(subset[[duration_col, event_col, '__grp__']], duration_col=duration_col, event_col=event_col)
                hr = float(cph_tmp.hazard_ratios_.values[0])
                summ = cph_tmp.summary
                hr_lower = float(summ['exp(coef) lower 95%'].iloc[0])
                hr_upper = float(summ['exp(coef) upper 95%'].iloc[0])
            except Exception:
                pass
            results[f"{cat1} vs {cat2}"] = {'p_value': p, 'HR': hr, 'HR_lower': hr_lower, 'HR_upper': hr_upper}
    return results

def fit_cox_for_var_no_cache(df, duration_col, event_col, var, include_numeric=False):
    d = df[[duration_col, event_col, var]].copy()
    dummies = pd.get_dummies(d[var], prefix=var.replace(' ', '_'), drop_first=True)
    X = pd.concat([d[[duration_col, event_col]].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    if include_numeric and 'Monthly Charges' in df.columns:
        try:
            X = pd.concat([X, df.loc[X.index, ['Monthly Charges']].reset_index(drop=True)], axis=1)
        except Exception:
            pass
    X = X.dropna()
    try:
        cph = CoxPHFitter()
        cph.fit(X, duration_col=duration_col, event_col=event_col)
        hr = cph.hazard_ratios_.to_frame(name='HR')
        concord = cph.concordance_index_
        return cph, hr, concord, None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data
def fit_cox_for_var(df, duration_col, event_col, var, include_numeric=False):
    d = df[[duration_col, event_col, var]].copy()
    # Codificar variable categórica con one-hot
    dummies = pd.get_dummies(d[var], prefix=var.replace(' ', '_'), drop_first=True)
    X = pd.concat([d[[duration_col, event_col]].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    # Opcionalmente incluir Monthly Charges si está presente y se solicita
    if include_numeric and 'Monthly Charges' in df.columns:
        try:
            X = pd.concat([X, df.loc[X.index, ['Monthly Charges']].reset_index(drop=True)], axis=1)
        except Exception:
            pass
    # Eliminar NA y asegurar tipo numérico
    X = X.dropna()
    try:
        cph = CoxPHFitter()
        cph.fit(X, duration_col=duration_col, event_col=event_col)
        hr = cph.hazard_ratios_.to_frame(name='HR')
        concord = cph.concordance_index_
        return cph, hr, concord, None
    except Exception as e:
        # Devolver el texto de la excepción para explicar por qué falló el modelo
        return None, None, None, str(e)

@st.cache_resource
def fit_cox_time_interaction_model(df, selected_vars=None):
    # Modelo Cox con interacciones temporales para predicciones
    dur_col = 'Tenure Months'
    evt_col = 'Churn Value'

    default_features = [
        'Gender', 'Senior Citizen', 'Partner', 'Dependents',
        'Phone Service', 'Contract', 'Internet Service',
        'Paperless Billing', 'Payment Method'
    ]

    if selected_vars:
        categorical_features = [c for c in selected_vars if c in df.columns]
    else:
        categorical_features = default_features

    # Filtrar columnas categóricas disponibles
    categorical_features = [c for c in categorical_features if c in df.columns]

    # Base con duración/evento
    df_base = df[[dur_col, evt_col]].copy()
    df_base[dur_col] = pd.to_numeric(df_base[dur_col], errors='coerce')
    df_base[evt_col] = pd.to_numeric(df_base[evt_col], errors='coerce')

    # Dummies
    if categorical_features:
        df_dummies = pd.get_dummies(df[categorical_features], drop_first=True)
    else:
        df_dummies = pd.DataFrame(index=df.index)
    df_model = pd.concat([df_base, df_dummies], axis=1)

    # Filtrar filas válidas
    df_model = df_model[(df_model[dur_col] > 0) & (df_model[evt_col].isin([0, 1]))].dropna()

    # Crear log_t
    df_model['log_t'] = np.log(df_model[dur_col].clip(lower=0.1))

    # Interacciones con tiempo (si existen columnas)
    for base_col in ['Contract_One year', 'Contract_Two year']:
        if base_col in df_model.columns:
            inter_col = f"{base_col}_logt"
            df_model[inter_col] = pd.to_numeric(df_model[base_col], errors='coerce').fillna(0).astype(int) * df_model['log_t']

    # Ajuste del modelo
    cph_td = CoxPHFitter(penalizer=0.01)
    cph_td.fit(df_model, duration_col=dur_col, event_col=evt_col)
    return cph_td

def build_feature_map_from_model_cols(model_cols):
    fmap = []
    for col in model_cols:
        col = str(col)
        if col == 'log_t' or col.endswith('_logt'):
            fmap.append((col, None))
            continue
        if ('_' in col or ' ' in col) and not col.replace('_','').replace(' ','').replace('.','').isdigit():
            if '_' in col:
                parts = col.rsplit('_', 1)
            else:
                parts = col.rsplit(' ', 1)
            feat, cat = parts[0], parts[1]
            fmap.append((feat, cat))
        else:
            fmap.append((col, None))
    return fmap

def profile_to_vector_from_model(profile_dict, feature_map):
    x = []
    for feat, cat in feature_map:
        if cat is None:
            x.append(float(profile_dict.get(feat, 0.0)))
        else:
            key_exact = f"{feat}_{cat}"
            if key_exact in profile_dict:
                x.append(float(profile_dict.get(key_exact, 0.0)))
            else:
                val = profile_dict.get(feat)
                if val is None:
                    x.append(0.0)
                else:
                    if isinstance(val, (int, float)):
                        x.append(1.0 if float(val) == 1.0 else 0.0)
                    else:
                        x.append(1.0 if str(val) == str(cat) else 0.0)
    return np.array(x, dtype=float)

def interp_baseline_H0_for_model(cph, t):
    H0_df = cph.baseline_cumulative_hazard_.copy()
    if isinstance(H0_df, pd.Series):
        s = H0_df.astype(float).copy()
    else:
        s = H0_df.iloc[:, 0].astype(float).copy()
    times = s.index.values.astype(float)
    vals = s.values.astype(float).squeeze()
    return float(np.interp(t, times, vals))

def compute_survival_curve_for_profile(cph, profile_dict, t_grid):
    model_cols = list(cph.params_.index)
    fmap = build_feature_map_from_model_cols(model_cols)
    X = profile_to_vector_from_model(profile_dict, fmap)

    # rellenar log_t si existe
    if 'log_t' in model_cols:
        tenure = float(profile_dict.get('Tenure Months', profile_dict.get('tenure', 0.1)))
        X[model_cols.index('log_t')] = math.log(max(tenure, 0.1))

    # manejar columnas *_logt
    for i, col in enumerate(model_cols):
        if str(col).endswith('_logt'):
            base_name = str(col)[:-5]
            if X[i] == 0.0:
                key_alt = base_name
                key_alt2 = base_name.replace(' ', '_')
                base_val = float(profile_dict.get(key_alt, profile_dict.get(key_alt2, 0.0)))
                if 'log_t' in model_cols:
                    logt_val = X[model_cols.index('log_t')]
                else:
                    logt_val = math.log(max(float(profile_dict.get('Tenure Months', 0.1)), 0.1))
                X[i] = base_val * logt_val

    beta = cph.params_.values.astype(float).squeeze()
    exp_factor = float(np.exp(float(np.dot(beta, X))))
    Lambda0_grid = np.array([interp_baseline_H0_for_model(cph, t) for t in t_grid])
    Lambda_grid = Lambda0_grid * exp_factor
    S_grid = np.exp(-Lambda_grid)
    return S_grid

def prob_conditional_interval(model, profile_dict, t1, t2):
    model_cols = list(model.params_.index)
    fmap = build_feature_map_from_model_cols(model_cols)
    X = profile_to_vector_from_model(profile_dict, fmap)
    beta = model.params_.values.astype(float).squeeze()
    exp_factor = float(np.exp(np.dot(beta, X)))

    Lam1_0 = interp_baseline_H0_for_model(model, t1)
    Lam2_0 = interp_baseline_H0_for_model(model, t2)

    Lam1 = Lam1_0 * exp_factor
    Lam2 = Lam2_0 * exp_factor

    prob_cond = 1.0 - math.exp(-(Lam2 - Lam1))
    return prob_cond

def generate_dynamic_descriptions(selected_vars):
    """Genera descripciones personalizadas basadas en variables seleccionadas."""
    var_descriptions = {}
    
    # Mapeo de variables a sus características en cada perfil
    var_details = {
        'Contract': ('contrato de 2 años', 'contrato mes a mes'),
        'Internet Service': ('sin fibra óptica', 'fibra óptica'),
        'Payment Method': ('pago automático', 'pago con cheque electrónico'),
        'Online Security': ('con seguridad en línea', 'sin servicios de seguridad'),
        'Tech Support': ('con soporte técnico', 'sin soporte técnico'),
        'Online Backup': ('con respaldo en línea', 'sin respaldo'),
        'Device Protection': ('con protección de dispositivo', 'sin protección'),
        'Partner': ('con pareja', 'sin pareja'),
        'Dependents': ('con dependientes', 'sin dependientes'),
        'Phone Service': ('con servicio telefónico', 'sin servicio telefónico'),
        'Paperless Billing': ('sin facturación digital', 'con facturación digital'),
        'Streaming Movies': ('sin películas en streaming', 'con películas en streaming'),
        'Streaming TV': ('sin TV en streaming', 'con TV en streaming'),
        'Senior Citizen': ('no es adulto mayor', 'es adulto mayor'),
        'Multiple Lines': ('sin múltiples líneas', 'con múltiples líneas')
    }
    
    # Crear descripción dinámica del Perfil Bueno
    bueno_features = [var_details[v][0] for v in selected_vars if v in var_details]
    if bueno_features:
        features_text = ', '.join(bueno_features)
        var_descriptions['bueno'] = f"📈 **El cliente que queremos retener:** Tiene **{features_text}**, lleva 12 meses con nosotros y todo funciona perfecto. La línea verde ascendente muestra su historia de éxito: **sigue activo mes tras mes**. Los marcadores te dicen exactamente qué % permanece en cada hito (5, 12, 24, 48 meses)."
    else:
        var_descriptions['bueno'] = "📈 **El cliente que queremos retener:** Tiene todas las características positivas. La línea verde ascendente muestra su historia de éxito: **sigue activo mes tras mes**. Los marcadores te dicen exactamente qué % permanece en cada hito."
    
    # Crear descripción dinámica del Perfil Malo
    malo_features = [var_details[v][1] for v in selected_vars if v in var_details]
    if malo_features:
        features_text = ', '.join(malo_features)
        var_descriptions['malo'] = f"⚠️ **El cliente en riesgo:** Es nuevo (1 mes) y tiene **{features_text}**. Esto es un combo peligroso. La línea roja desciende rápidamente: al mes 24 solo le quedan ~20%. Los marcadores muestran la **caída acelerada**—estos clientes se van rápido y hay poco tiempo para actuar."
    else:
        var_descriptions['malo'] = "⚠️ **El cliente en riesgo:** Es nuevo (1 mes) con características negativas. Esto es un combo peligroso. La línea roja desciende rápidamente: al mes 24 solo le quedan ~20%. Los marcadores muestran la **caída acelerada**."
    
    # Descripción del Reloj del Churn - más amigable
    vars_text = ', '.join(selected_vars[:3]) if selected_vars else 'las variables seleccionadas'
    var_descriptions['reloj'] = (
        f"⏰ **Reloj de churn (riesgo mes a mes)**: Este gráfico muestra el **riesgo de abandono en el próximo mes** "
        f"*solo para clientes que siguen activos hasta hoy*. Las predicciones se calculan usando **{vars_text}** "
        f"(y sus categorías) para construir el perfil. **Picos** = meses con mayor peligro de churn; "
        f"**valles** = meses relativamente seguros. Úsalo para decidir **cuándo** intervenir con ofertas o seguimiento."
    )
    
    # Descripción de Comparación - más visual
    var_descriptions['comparacion'] = f"🔄 **Lado a lado:** Rojo vs Verde. El rojo (alto riesgo) oscila entre 15-30% de abandono mensual los primeros meses. El verde (bajo riesgo) se mantiene por debajo del 5%. Esta **brecha visible** te muestra exactamente el impacto de {', '.join(selected_vars[:2]) if selected_vars else 'tus variables'} en la retención."
    
    return var_descriptions

def build_dynamic_profiles(selected_vars, df):
    """Construye perfiles Bueno y Malo dinámicamente basados en variables seleccionadas."""
    
    # Mapeo de variables a sus categorías favorables (bueno) y desfavorables (malo)
    var_mapping = {
        'Contract': {
            'bueno': {'Contract_Two year': 1, 'Contract_One year': 0, 'Contract_Month-to-month': 0},
            'malo': {'Contract_Month-to-month': 1, 'Contract_One year': 0, 'Contract_Two year': 0}
        },
        'Internet Service': {
            'bueno': {'Internet Service_DSL': 1, 'Internet Service_Fiber optic': 0, 'Internet Service_No': 0},
            'malo': {'Internet Service_Fiber optic': 1, 'Internet Service_DSL': 0, 'Internet Service_No': 0}
        },
        'Payment Method': {
            'bueno': {'Payment Method_Credit card (automatic)': 1, 'Payment Method_Bank transfer (automatic)': 0, 
                     'Payment Method_Electronic check': 0, 'Payment Method_Mailed check': 0},
            'malo': {'Payment Method_Electronic check': 1, 'Payment Method_Credit card (automatic)': 0,
                    'Payment Method_Bank transfer (automatic)': 0, 'Payment Method_Mailed check': 0}
        },
        'Online Security': {
            'bueno': {'Online Security_Yes': 1, 'Online Security_No': 0},
            'malo': {'Online Security_No': 1, 'Online Security_Yes': 0}
        },
        'Tech Support': {
            'bueno': {'Tech Support_Yes': 1, 'Tech Support_No': 0},
            'malo': {'Tech Support_No': 1, 'Tech Support_Yes': 0}
        },
        'Online Backup': {
            'bueno': {'Online Backup_Yes': 1, 'Online Backup_No': 0},
            'malo': {'Online Backup_No': 1, 'Online Backup_Yes': 0}
        },
        'Device Protection': {
            'bueno': {'Device Protection_Yes': 1, 'Device Protection_No': 0},
            'malo': {'Device Protection_No': 1, 'Device Protection_Yes': 0}
        },
        'Partner': {
            'bueno': {'Partner_Yes': 1, 'Partner_No': 0},
            'malo': {'Partner_No': 1, 'Partner_Yes': 0}
        },
        'Dependents': {
            'bueno': {'Dependents_Yes': 1, 'Dependents_No': 0},
            'malo': {'Dependents_No': 1, 'Dependents_Yes': 0}
        },
        'Phone Service': {
            'bueno': {'Phone Service_Yes': 1, 'Phone Service_No': 0},
            'malo': {'Phone Service_No': 1, 'Phone Service_Yes': 0}
        },
        'Paperless Billing': {
            'bueno': {'Paperless Billing_No': 1, 'Paperless Billing_Yes': 0},
            'malo': {'Paperless Billing_Yes': 1, 'Paperless Billing_No': 0}
        },
        'Streaming Movies': {
            'bueno': {'Streaming Movies_No': 1, 'Streaming Movies_Yes': 0},
            'malo': {'Streaming Movies_Yes': 1, 'Streaming Movies_No': 0}
        },
        'Streaming TV': {
            'bueno': {'Streaming TV_No': 1, 'Streaming TV_Yes': 0},
            'malo': {'Streaming TV_Yes': 1, 'Streaming TV_No': 0}
        },
        'Senior Citizen': {
            'bueno': {'Senior Citizen_No': 1, 'Senior Citizen_Yes': 0},
            'malo': {'Senior Citizen_Yes': 1, 'Senior Citizen_No': 0}
        },
        'Multiple Lines': {
            'bueno': {'Multiple Lines_No': 1, 'Multiple Lines_Yes': 0},
            'malo': {'Multiple Lines_Yes': 1, 'Multiple Lines_No': 0}
        }
    }
    
    # Inicializar perfiles base
    profile_bueno = {'Tenure Months': 12.0}
    profile_malo = {'Tenure Months': 1.0}
    
    # Agregar solo las variables seleccionadas
    for var in selected_vars:
        if var in var_mapping:
            profile_bueno.update(var_mapping[var]['bueno'])
            profile_malo.update(var_mapping[var]['malo'])
    
    return profile_bueno, profile_malo

def render_cox_predictions_section(df, selected_vars=None):
    st.subheader("Predicciones (COX)")
    
    # Preparar descripciones personalizadas
    if selected_vars is None or len(selected_vars) == 0:
        selected_vars = ['Contract', 'Internet Service', 'Payment Method', 'Partner', 'Dependents']
    selected_vars = list(dict.fromkeys([v for v in selected_vars if v in df.columns]))
    if not selected_vars:
        selected_vars = ['Contract', 'Internet Service', 'Payment Method', 'Partner', 'Dependents']
    
    descriptions = generate_dynamic_descriptions(selected_vars)
    
    st.write(f"Con COX, analizamos cómo las variables {f'**({', '.join(selected_vars[:4])}{'...' if len(selected_vars) > 4 else ''})** ' if selected_vars else ''}afectan la probabilidad de que un cliente abandone. Aquí comparamos dos perfiles extremos: uno de bajo riesgo (cliente ideal) y otro de alto riesgo (cliente vulnerable).")
    with st.expander("📖 Cómo leer estas gráficas", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📈 Líneas más altas = Mejor retención**
            - S(t) es la probabilidad de que el cliente siga activo
            - Verde (bueno): cliente mantiene alto porcentaje
            - Rojo (malo): cliente abandona rápidamente
            """)
        with col2:
            st.markdown("""
            **⏱️ Los marcadores (puntos) son hitos importantes**
            - 5, 12, 24 meses: checkpoints críticos
            - Te muestran exactamente qué porcentaje permanece
            - Puedes comparar perfiles mes a mes
            """)

    try:
        cph_td = fit_cox_time_interaction_model(df, selected_vars=tuple(selected_vars))
    except Exception as e:
        st.warning(f"No se pudo ajustar el modelo de predicciones COX: {e}")
        return

    # Construir perfiles dinámicos basados en variables seleccionadas
    profile_bueno, profile_malo = build_dynamic_profiles(selected_vars, df)

    t_grid = np.linspace(0, 72, 721)
    S_bueno = compute_survival_curve_for_profile(cph_td, profile_bueno, t_grid)
    S_malo = compute_survival_curve_for_profile(cph_td, profile_malo, t_grid)

    # Curva de supervivencia - Perfil Bueno
    st.markdown("#### 🟢 Perfil Bueno (Cliente Ideal)")
    st.caption(descriptions['bueno'])
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='black')
    ax.set_facecolor('black')
    ax.plot(t_grid, S_bueno, color='green', linewidth=3, label='S(t) perfil bueno')
    ax.fill_between(t_grid, S_bueno, color='green', alpha=0.08)
    for m in [5, 12, 24, 48]:
        s_m = float(np.interp(m, t_grid, S_bueno))
        ax.plot(m, s_m, 'o', color='#ff7f0e', markersize=8, markeredgecolor='white', markeredgewidth=1.2)
        ax.text(m, s_m - 0.04, f"{m}m: {s_m*100:.2f}%", color='#ff7f0e', fontsize=9, ha='center', va='top')
    ax.set_xlim(0, 72)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Tiempo (meses)", color='white')
    ax.set_ylabel("Probabilidad de supervivencia S(t)", color='white')
    ax.set_title("Supervivencia estimada", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    st.pyplot(fig)
    plt.close(fig)

    # Curva de supervivencia - Perfil Malo
    st.markdown("#### 🔴 Perfil Malo (Cliente en Riesgo)")
    st.caption(descriptions['malo'])
    fig, ax = plt.subplots(figsize=(8, 4), facecolor='black')
    ax.set_facecolor('black')
    ax.plot(t_grid, S_malo, color='red', linewidth=3, label='S(t) perfil malo')
    ax.fill_between(t_grid, S_malo, color='red', alpha=0.08)
    for m in [5, 12, 24]:
        s_m = float(np.interp(m, t_grid, S_malo))
        ax.plot(m, s_m, 'o', color='red', markersize=8, markeredgecolor='white', markeredgewidth=1.2)
        ax.text(m, s_m - 0.04, f"{m}m: {s_m*100:.2f}%", color='red', fontsize=9, ha='center', va='top')
    ax.set_xlim(0, 72)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Tiempo (meses)", color='white')
    ax.set_ylabel("Probabilidad de supervivencia S(t)", color='white')
    ax.set_title("Supervivencia estimada", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.2, color='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    st.pyplot(fig)
    plt.close(fig)

    # Reloj del Churn: probabilidad condicional por mes (perfil malo)
    st.markdown("#### ⏰ Reloj del Churn — Mes a Mes")
    st.caption(descriptions['reloj'])
    month_names = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    t_values = np.arange(1, 13)
    probs = [prob_conditional_interval(cph_td, profile_malo, t, t+1) for t in t_values]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), facecolor='black')
    ax.set_facecolor('black')
    ax.plot(t_values, probs, marker='o', linestyle='-', color='cyan')
    ax.set_title("Probabilidad condicional de churn por mes", color='white')
    ax.set_xlabel("Mes", color='white')
    ax.set_ylabel("Probabilidad condicional", color='white')
    ax.set_xticks(t_values)
    ax.set_xticklabels(month_names[:len(t_values)], rotation=45, ha='right', color='white', fontsize=8)
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, color='white')
    fig.subplots_adjust(bottom=0.28)
    st.pyplot(fig)
    plt.close(fig)


    # Comparación de perfiles (malo vs bueno)
    st.markdown("#### ⚖️ Comparación: ¿Cuál es la diferencia?")
    st.caption(descriptions['comparacion'])
    t_values = np.arange(1, 13)
    probs_malo = [prob_conditional_interval(cph_td, profile_malo, t, t+1) for t in t_values]
    probs_bueno = [prob_conditional_interval(cph_td, profile_bueno, t, t+1) for t in t_values]
    fig, ax = plt.subplots(figsize=(6.4, 2.8), facecolor='black')
    ax.set_facecolor('black')
    ax.plot(t_values, probs_malo, marker='o', linestyle='-', color='red', label='Perfil alto riesgo')
    ax.plot(t_values, probs_bueno, marker='o', linestyle='-', color='green', label='Perfil bajo riesgo')
    ax.set_title("Reloj del Churn — Comparación de perfiles", color='white')
    ax.set_xlabel("Intervalo (mes t → t+1)", color='white')
    ax.set_ylabel("Probabilidad condicional de churn", color='white')
    ax.set_xticks(t_values)
    ax.set_xticklabels(month_names[:len(t_values)], rotation=45, ha='right', color='white', fontsize=8)
    ax.tick_params(colors='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.3, color='white')
    fig.subplots_adjust(bottom=0.28)
    st.pyplot(fig)
    plt.close(fig)

    # Incremento porcentual mes a mes (perfil malo)
    st.markdown("#### 📊 Volatilidad del Riesgo (Cambios Mes a Mes)")
    st.caption("Analiza la **volatilidad del riesgo**: cambios porcentuales mes a mes en la probabilidad de churn. Valores positivos indican aceleración del riesgo; negativos, desaceleración. Útil para identificar cuándo el riesgo se estabiliza o crece exponencialmente.")
    increments = []
    for i in range(1, len(probs)):
        if probs[i-1] > 0:
            inc = (probs[i] - probs[i-1]) / probs[i-1] * 100
        else:
            inc = np.nan
        increments.append(inc)
    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor='black')
    ax.set_facecolor('black')
    ax.plot(range(2, len(probs)+1), increments, marker='o', linestyle='-', color='magenta')
    ax.set_title("Incremento porcentual de la probabilidad condicional de churn", color='white')
    ax.set_xlabel("Intervalo (mes t → t+1)", color='white')
    ax.set_ylabel("Incremento porcentual (%)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, alpha=0.3, color='white')
    st.pyplot(fig)
    plt.close(fig)

    # Gráfico polar (perfil malo)
    st.markdown("#### 🔄 Vista Circular del Riesgo (Reloj Visual)")
    st.caption("**Visualización cíclica** del riesgo: cada mes ocupa una posición en el \"reloj\". La distancia desde el centro = riesgo. Detecta patrones estacionales o cíclicos en abandonos. Para este perfil, los meses iniciales (1-3) están más alejados, indicando riesgo máximo al inicio.")
    month_names = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    slider_col, pct_col = st.columns([3, 1])
    with slider_col:
        selected_month = st.slider("🕒 Línea de tiempo (mes)", min_value=1, max_value=len(probs), value=1, step=1)
    with pct_col:
        sel_idx = selected_month - 1
        pct = probs[sel_idx] * 100 if len(probs) > 0 else 0
        st.metric("Riesgo próximo mes", f"{pct:.2f}%")
        st.caption("Probabilidad condicional de churn")
    months = np.arange(1, len(probs)+1)
    angles = np.linspace(0, 2*np.pi, len(probs), endpoint=False)
    fig = plt.figure(figsize=(4, 4), facecolor='black')
    ax = plt.subplot(111, polar=True, facecolor='black')
    ax.plot(angles, probs, marker='o', linestyle='-', color='cyan')
    ax.fill(angles, probs, alpha=0.3, color='cyan')

    # Aguja según mes seleccionado
    sel_angle = angles[sel_idx]
    sel_radius = probs[sel_idx]
    ax.plot([sel_angle, sel_angle], [0, sel_radius], color='yellow', linewidth=2)
    ax.scatter([sel_angle], [sel_radius], color='yellow', s=80, edgecolor='white', linewidth=1.2, zorder=5)
    if len(probs) <= 12:
        ax.set_xticks(angles)
        ax.set_xticklabels([month_names[(m - 1) % 12] for m in months], color='white', fontsize=8)
    else:
        tick_idx = np.arange(0, len(probs), 6)
        ax.set_xticks(angles[tick_idx])
        ax.set_xticklabels([str(months[i]) for i in tick_idx], color='white', fontsize=8)
    ax.set_title("Reloj del Churn (Probabilidad Condicional)", va='bottom', color='white')
    ax.tick_params(colors='white')
    ax.spines['polar'].set_color('white')
    ax.grid(color='white', alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# --- Interfaz de Usuario ---
# CSS para reducir ancho del sidebar
st.markdown("""
<style>
    /* Reducir ancho del sidebar al 80% (de 37rem a 30rem) */
    [data-testid="stSidebar"] {
        min-width: 30rem;
        max-width: 30rem;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 30rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Retención Inteligente: Anticipando el Churn</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
df = load_data()

# Sidebar: selección de variables significativas (preseleccionadas)
with st.sidebar:
    st.header("Variables significativas")
    recommended = ['Contract', 'Online Security', 'Tech Support', 'Payment Method',
                   'Online Backup', 'Device Protection', 'Partner', 'Dependents',
                   'Internet Service', 'Streaming Movies', 'Streaming TV',
                   'Paperless Billing', 'Senior Citizen', 'Multiple Lines']

    # Grupos de columnas que NO son predictores
    id_cols = ['CustomerID']
    geo_cols = ['Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude']
    # 'Churn Value' se usa como variable de evento, no como predictor
    non_predictor_cols = ['Count', 'Churn Reason', 'Churn Label', 'Churn Score', 'CLTV', 'Churn Value']
    disabled_cols = set(id_cols + geo_cols + non_predictor_cols)

    # Construir la lista de variables seleccionables que SÍ se usan en los modelos
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Solo mostrar las variables recomendadas (si están presentes y no deshabilitadas)
    available = [c for c in recommended if c in obj_cols and c not in disabled_cols]

    # Asegurar unicidad y preservar orden
    available = list(dict.fromkeys(available))
    default_sel = [x for x in recommended[:3] if x in available]

    selected = st.multiselect("Seleccion de variables:", options=available, default=default_sel, key="selected_vars")
    selected_tuple = tuple(selected)
    prev_selected = st.session_state.get("selected_vars_prev")
    if prev_selected != selected_tuple:
        st.session_state["selected_vars_prev"] = selected_tuple
        st.session_state["cox_ready"] = False
        st.session_state["cox_ready_set"] = False
        try:
            compute_km.clear()
            compute_logrank_pairs.clear()
            fit_cox_for_var.clear()
        except Exception:
            pass
    st.caption("Sugerencia: selecciona una sola variable para análisis más claro, o varias para comparar múltiples variables una a una")

    st.markdown("---")
    st.write("Instrucciones:")
    st.write("Para cada variable seleccionada, verás las curvas o pruebas de supervivencia siguientes, según corresponda:")
    st.markdown("• Kaplan-Meier\n• Log-Rank\n• Regresión COX")
    include_numeric = st.checkbox("Incluir 'Monthly Charges' en el modelo COX", value=True)

    # Panel colapsable para mostrar columnas no predictoras/identificadoras/geográficas solo bajo demanda
    with st.expander("Variables deshabilitadas", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write("**Identificadores**")
            for col in id_cols:
                if col in df.columns:
                    st.checkbox(col, value=False, disabled=True)
        with col_b:
            st.write("**Geográficas**")
            for col in geo_cols:
                if col in df.columns:
                    st.checkbox(col, value=False, disabled=True)
        with col_c:
            st.write("**No predictoras**")
            for col in non_predictor_cols:
                if col in df.columns:
                    st.checkbox(col, value=False, disabled=True)
        st.caption("Nota: columnas no predictoras para los modelos porque no muestran un impacto en el abandono de los clientes")

# Área principal: mostrar tabla y análisis por variable
st.subheader("Vista previa e inicial de los datos")
st.dataframe(df.head(5))

if not selected:
    st.warning("Selecciona al menos una variable en la barra lateral para ver los análisis.")
else:
    # Selector central único para todos los modelos
    choice_cols = st.columns([1,2,1])
    with choice_cols[1]:
        model_options = ["KM", "Log‑Rank", "Cox"]
        if st.session_state.get("cox_ready"):
            model_options.append("⭐ Predicciones")
        # Usar session_state para mantener la selección del usuario
        current_choice = st.session_state.get("view_choice_global", "KM")
        # Si la opción actual no está en model_options (ej: quitó "Predicciones"), resetear a KM
        if current_choice not in model_options:
            current_choice = "KM"
            st.session_state["view_choice_global"] = current_choice
        view_choice = st.radio("Seleccionar modelo:", options=model_options, index=model_options.index(current_choice), horizontal=True, key="view_choice_global")
    
    st.markdown("---")
    
    # Mostrar cuadro de información una sola vez según el modelo seleccionado
    if view_choice == "KM":
        with st.expander("ℹ️ ¿Qué es Kaplan-Meier?", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""<div style='padding-right: 15px; border-right: 2px solid #ddd;'>
                <strong>Kaplan-Meier</strong> es un análisis que te muestra:<br><br>
                📊 <strong>¿Cuántos clientes permanecen activos con el tiempo?</strong>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown("""<div style='padding-left: 15px; padding-right: 15px; border-right: 2px solid #ddd;'>
                <strong>Visualiza:</strong><br>
                • Porcentaje de retención mes a mes<br>
                • Compara grupos de clientes
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown("""<div style='padding-left: 15px;'>
                <strong>Identifica:</strong><br>
                • Cuándo ocurre la mayor pérdida<br>
                • Grupos con mejor/peor retención
                </div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**¿Qué significa esta gráfica?**")
        st.write("La curva de Kaplan-Meier muestra el **porcentaje de clientes que permanecen activos** a lo largo del tiempo. Entre más alta esté la línea, más clientes se quedan con el servicio.")
    elif view_choice == "Cox":
        with st.expander("ℹ️ Cox Proportional Hazards (COX)", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""<div style='padding-right: 15px; border-right: 2px solid #ddd;'>
                <strong>Modelo Cox</strong> analiza el <strong>riesgo de abandono</strong> considerando múltiples factores.<br><br>
                💡 Identifica qué factores aumentan o disminuyen el riesgo de churn.
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown("""<div style='padding-left: 15px; padding-right: 15px; border-right: 2px solid #ddd;'>
                📈 <strong>Hazard Ratio (HR)</strong><br>
                • <strong>HR > 1</strong>: ⬆️ Aumenta riesgo instantaneo<br>
                • <strong>HR < 1</strong>: ⬇️ Reduce riesgo instantaneo<br>
                • <strong>HR = 1</strong>: Sin efecto
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown("""<div style='padding-left: 15px;'>
                🎯 <strong>C-index</strong> (Precisión)<br>
                • Valor: 0.5 a 1.0<br>
                • > 0.7 = Excelente<br>
                • > 0.6 = Bueno
                </div>""", unsafe_allow_html=True)
    elif view_choice == "Log‑Rank":
        with st.expander("ℹ️ Log‑Rank (comparaciones por pares)", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""<div style='padding-right: 15px; border-right: 2px solid #ddd;'>
                <strong>Test Log-Rank</strong> compara curvas de supervivencia entre grupos.<br><br>
                🔍 ¿Hay diferencias reales entre grupos?
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown("""<div style='padding-left: 15px; padding-right: 15px; border-right: 2px solid #ddd;'>
                <strong>Compara:</strong><br>
                • Cada par de categorías<br>
                • Diferencias estadísticamente significativas<br>
                • Porcentaje de diferencia en riesgo
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown("""<div style='padding-left: 15px;'>
                <strong>Interpretación:</strong><br>
                • ✅ p-value < 0.05: Diferencia significativa<br>
                • ❌ p-value ≥ 0.05: Sin diferencia clara
                </div>""", unsafe_allow_html=True)
    
    st.markdown("---")

    if view_choice == "KM":
        st.markdown("### Kaplan–Meier (KM) — Curva general")
        st.caption("Gráfica actualizada con las variables seleccionadas")
        try:
            plotted = False
            summaries = []
            last_surv = []
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                for var in selected:
                    km_results = compute_km(df, 'Tenure Months', 'Churn Value', var)
                    for grp, res in km_results.items():
                        if res is None or res.empty:
                            continue
                        label = f"{var}: {grp}"
                        fig.add_trace(go.Scatter(
                            x=res['timeline'], y=res['survival'], mode='lines', name=label,
                            hovertemplate=f'{label}<br>Mes: %{{x}}<br>Supervivencia: %{{y:.3f}}<extra></extra>'
                        ))
                        plotted = True
                        surv = res['survival']
                        timeline = res['timeline']
                        median_time = None
                        if (surv <= 0.5).any():
                            median_time = timeline[surv <= 0.5].iloc[0]
                        last_surv.append((label, float(surv.iloc[-1]), float(timeline.iloc[-1])))
                        if median_time is None:
                            summaries.append(f"**{label}**: la mayoría de clientes permanece activa durante todo el periodo analizado.")
                        else:
                            summaries.append(f"**{label}**: aproximadamente la mitad abandona después de {median_time:.0f} meses.")

                if not plotted:
                    st.markdown("<h3 style='color:red'>No se pudo ajustar Kaplan–Meier: datos insuficientes</h3>", unsafe_allow_html=True)
                else:
                    fig.update_layout(
                        title='Curvas KM (variables seleccionadas)',
                        xaxis_title='Tiempo (meses)',
                        yaxis_title='Probabilidad de supervivencia',
                        template='plotly_white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(9, 5))
                for var in selected:
                    km_results = compute_km(df, 'Tenure Months', 'Churn Value', var)
                    for grp, res in km_results.items():
                        if res is None or res.empty:
                            continue
                        label = f"{var}: {grp}"
                        ax.step(res['timeline'], res['survival'], where='post', label=label)
                        plotted = True
                        surv = res['survival']
                        timeline = res['timeline']
                        median_time = None
                        if (surv <= 0.5).any():
                            median_time = timeline[surv <= 0.5].iloc[0]
                        last_surv.append((label, float(surv.iloc[-1]), float(timeline.iloc[-1])))
                        if median_time is None:
                            summaries.append(f"**{label}**: la mayoría de clientes permanece activa durante todo el periodo analizado.")
                        else:
                            summaries.append(f"**{label}**: aproximadamente la mitad abandona después de {median_time:.0f} meses.")

                if not plotted:
                    st.markdown("<h3 style='color:red'>No se pudo ajustar Kaplan–Meier: datos insuficientes</h3>", unsafe_allow_html=True)
                else:
                    ax.set_xlabel('Tiempo (meses)')
                    ax.set_ylabel('Probabilidad de supervivencia')
                    ax.set_title('Curvas KM (variables seleccionadas)')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close(fig)

            if plotted:
                st.markdown("**Interpretación:**")
                if last_surv:
                    best = sorted(last_surv, key=lambda x: x[1], reverse=True)[0]
                    worst = sorted(last_surv, key=lambda x: x[1])[0]
                    st.success(f"✅ **Mejor retención**: {best[0]} — al final del periodo aún {best[1]*100:.0f}% de clientes siguen activos.")
                    if best[0] != worst[0]:
                        st.error(f"⚠️ **Menor retención**: {worst[0]} — al final del periodo solo {worst[1]*100:.0f}% permanece.")
                st.markdown("---")
                for line in summaries[:6]:
                    st.write(f"• {line}")
        except Exception as e:
            st.markdown(f"<h3 style='color:red'>No se pudo ajustar Kaplan–Meier</h3>\n\n**Razón:** {e}", unsafe_allow_html=True)
    elif view_choice == "⭐ Predicciones":
        render_cox_predictions_section(df, selected)
    else:
        for var in selected:
            st.markdown(f"### Variable: **{var}**")

            def render_cox():
                st.write("**Cox Proportional Hazards (COX)**")
                
                # Comparación automática: Con vs Sin Monthly Charges
                if 'Monthly Charges' in df.columns:
                    st.markdown(f"#### 🔬 Comparación: Impacto de incluir 'Monthly Charges' — {var}")
                    
                    # Ajustar ambos modelos (usar versión sin caché para que se recalcule cada vez)
                    cph_sin, hr_sin, concord_sin, err_sin = fit_cox_for_var_no_cache(df, 'Tenure Months', 'Churn Value', var, include_numeric=False)
                    cph_con, hr_con, concord_con, err_con = fit_cox_for_var_no_cache(df, 'Tenure Months', 'Churn Value', var, include_numeric=True)
                    
                    if cph_sin is not None and cph_con is not None and concord_sin is not None and concord_con is not None:
                        col_comp1, col_comp2, col_comp3 = st.columns(3)
                        
                        with col_comp1:
                            st.metric("📊 Sin Monthly Charges", f"{concord_sin*100:.2f}%", 
                                     help="Precisión del modelo usando solo variables categóricas")
                        
                        with col_comp2:
                            st.metric("💰 Con Monthly Charges", f"{concord_con*100:.2f}%",
                                     help="Precisión del modelo incluyendo el precio mensual")
                        
                        with col_comp3:
                            diff = (concord_con - concord_sin) * 100
                            delta_color = "normal" if abs(diff) < 1 else "off" if diff < 0 else "normal"
                            st.metric("📈 Mejora", f"{diff:+.2f}%", 
                                     delta=f"{'Mejor' if diff > 0 else 'Peor' if diff < 0 else 'Igual'}", 
                                     delta_color=delta_color)
                        
                        # Interpretación automática
                        st.markdown("---")
                        if abs(diff) < 0.5:
                            st.info(f"💡 **Interpretación ({var}):** Monthly Charges tiene **poco impacto** en la precisión. "
                                    "Las variables categóricas ya explican casi todo. Puedes omitirla para un modelo más simple.")
                        elif diff > 2:
                            st.success("💡 **Interpretación ({}):** Monthly Charges **mejora significativamente** el modelo (+{:.2f}%). "
                                      "El precio es un factor importante independiente de los servicios contratados.".format(var, diff))
                        elif diff > 0.5:
                            st.success("💡 **Interpretación ({}):** Monthly Charges **mejora moderadamente** el modelo (+{:.2f}%). "
                                      "Incluirla da una ventaja, pero las variables categóricas siguen siendo dominantes.".format(var, diff))
                        else:
                            st.warning("💡 **Interpretación ({}):** Monthly Charges **empeora** el modelo ({:.2f}%). "
                                      "Probablemente hay multicolinealidad. Mejor usar solo variables categóricas.".format(var, diff))
                            
                            # Expander con título único por variable para evitar reutilización de widgets
                            with st.expander(f"📊 Multicolinealidad y variables categóricas — {var} ({diff:.2f}%)"):
                                st.markdown("""
                                **Multicolinealidad:** Cuando dos o más variables están altamente correlacionadas, 
                                el modelo no puede distinguir cuál es el verdadero factor causante.
                                """)
                                
                                # Obtener categorías únicas de la variable seleccionada
                                if var in df.columns:
                                    categories = df[var].unique()
                                    st.markdown("**Variables categóricas en la variable '{}':**".format(var))
                                    for i, cat in enumerate(categories, 1):
                                        st.write(f"• {cat}")
                        st.markdown("---")
                
                # Continuar con el modelo seleccionado
                cph, hr, concord, err = fit_cox_for_var(df, 'Tenure Months', 'Churn Value', var, include_numeric=include_numeric)
                if cph is None:
                    reason = err or "No fue posible ajustar el modelo Cox con las columnas seleccionadas (p. ej. muy pocas observaciones o colinealidad)."
                    st.markdown(f"<h3 style='color:red'>Modelo COX no pudo ser obtenido</h3>\n\n**Razón:** {reason}", unsafe_allow_html=True)
                    return
                try:
                    st.session_state["cox_ready"] = True
                    if not st.session_state.get("cox_ready_set"):
                        st.session_state["cox_ready_set"] = True
                        st.rerun()
                    
                    st.markdown(f"#### 📋 Resultados del Modelo {'**CON** Monthly Charges' if include_numeric and 'Monthly Charges' in df.columns else '**SIN** Monthly Charges'}")
                    
                    summary = cph.summary
                    hr_df = summary[["exp(coef)"]].rename(columns={"exp(coef)": "HR"})

                    # Cambio porcentual y cadenas formateadas con texto explicativo
                    hr_pct_num = (hr_df['HR'] - 1) * 100
                    effect_text = hr_pct_num.apply(lambda x: "Aumenta riesgo" if x > 0 else "Reduce riesgo")
                    display_df = pd.DataFrame({
                        'Efecto': effect_text,
                        'Porcentaje': hr_pct_num.abs().map(lambda x: f"{x:.2f}%"),
                    }, index=hr_df.index)

                    # Métricas visuales
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        if concord is not None:
                            st.metric("🎯 Precisión del Modelo (C-index)", f"{concord*100:.2f}%")
                    with col_m2:
                        num_accel = len(hr_df[hr_df['HR'] > 1])
                        num_reduce = len(hr_df[hr_df['HR'] < 1])
                        st.metric("⚖️ Balance Factores", f"{num_accel} aumentan / {num_reduce} reducen")
                    
                    # Gráfico de Barras: Impacto % de cada categoría
                    st.markdown("#### 📊 Impacto de cada categoría en el abandono")
                    try:
                        # Preparar datos para el gráfico de barras
                        impact_df = hr_df.copy()
                        impact_df['Impacto (%)'] = (impact_df['HR'] - 1) * 100
                        impact_df = impact_df.sort_values('Impacto (%)', ascending=True).tail(10)  # Top 10
                        
                        fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
                        ax.set_facecolor('black')
                        
                        colors = ['#ff4444' if x > 0 else '#00ff88' for x in impact_df['Impacto (%)']]
                        bars = ax.barh(range(len(impact_df)), impact_df['Impacto (%)'], color=colors, alpha=0.8)
                        
                        # Añadir línea de referencia en 0
                        ax.axvline(x=0, color='white', linestyle='--', linewidth=2, alpha=0.7)
                        
                        # Etiquetas
                        ax.set_yticks(range(len(impact_df)))
                        ax.set_yticklabels(impact_df.index, color='white', fontsize=10)
                        ax.set_xlabel('Cambio en el riesgo de abandono (%)', color='white', fontsize=11)
                        ax.set_title('Top 10 Categorías por Impacto', color='white', fontsize=13, fontweight='bold')
                        ax.tick_params(colors='white')
                        ax.grid(axis='x', alpha=0.3, color='white')
                        
                        # Añadir valores en las barras
                        for i, (idx, value) in enumerate(impact_df['Impacto (%)'].items()):
                            x_pos = value + (3 if value > 0 else -3)
                            ha = 'left' if value > 0 else 'right'
                            ax.text(x_pos, i, f"{value:+.1f}%", va='center', ha=ha, color='white', fontweight='bold', fontsize=9)
                        
                        # Anotaciones
                        ax.text(0.98, 0.02, 'Rojo = Aumenta riesgo | Verde = Reduce riesgo', 
                               transform=ax.transAxes, ha='right', va='bottom', 
                               color='white', fontsize=9, style='italic', alpha=0.7)
                        
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"No se pudo generar gráfico de impacto: {e}")
                    
                    # Forest Plot mejorado (HR + IC 95%)
                    st.markdown("#### 🎯 Forest Plot - Hazard Ratios con Intervalos de Confianza")
                    st.caption("Cada punto muestra el HR (Hazard Ratio) con su intervalo de confianza al 95%. La línea vertical en 1.0 representa 'sin efecto'.")
                    try:
                        conf_ints = cph.confidence_intervals_
                        if conf_ints is not None and not conf_ints.empty:
                            ci_lower_col = conf_ints.columns[0]
                            ci_upper_col = conf_ints.columns[1]
                            plot_df = hr_df.join(conf_ints[[ci_lower_col, ci_upper_col]], how='left')
                            plot_df = plot_df.dropna().sort_values('HR', ascending=True).tail(10)  # Top 10

                            if not plot_df.empty:
                                fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
                                ax.set_facecolor('black')
                                
                                y_pos = range(len(plot_df))
                                for i, (name, row) in enumerate(plot_df.iterrows()):
                                    hr_val = float(row['HR'])
                                    lower = float(row[ci_lower_col])
                                    upper = float(row[ci_upper_col])
                                    color = '#ff4444' if hr_val > 1 else '#00ff88'
                                    
                                    # Línea del intervalo
                                    ax.plot([lower, upper], [i, i], color=color, linewidth=3, alpha=0.6)
                                    # Punto del HR
                                    ax.plot(hr_val, i, 'o', color=color, markersize=12, markeredgecolor='white', markeredgewidth=2)
                                    # Etiqueta con valor
                                    ax.text(upper + 0.1, i, f"HR={hr_val:.2f}", va='center', color='white', fontsize=9)
                                
                                # Línea de referencia
                                ax.axvline(x=1, color='yellow', linestyle='--', linewidth=2, alpha=0.8, label='Sin efecto (HR=1)')
                                
                                # Configuración
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels([str(n) for n in plot_df.index], color='white', fontsize=10)
                                ax.set_xlabel("Hazard Ratio (HR)", color='white', fontsize=11)
                                ax.set_title("Top 10 Categorías - Forest Plot", color='white', fontsize=13, fontweight='bold')
                                ax.tick_params(colors='white')
                                ax.grid(axis='x', alpha=0.3, color='white')
                                ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
                                
                                # Anotaciones explicativas (alineadas cerca del label del eje X)
                                ax.text(0.02, -0.12, '← Reduce riesgo', transform=ax.transAxes, 
                                       ha='left', va='top', color='#00ff88', fontsize=10, fontweight='bold', clip_on=False)
                                ax.text(0.98, -0.12, 'Aumenta riesgo →', transform=ax.transAxes, 
                                       ha='right', va='top', color='#ff4444', fontsize=10, fontweight='bold', clip_on=False)
                                
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.info("No hay suficientes datos para el Forest Plot.")
                        else:
                            st.info("No hay intervalos de confianza disponibles para el Forest Plot.")
                    except Exception as e:
                        st.warning(f"No se pudo generar Forest Plot: {e}")
                    
                    # Top covariables con interpretación simplificada
                    try:
                        top = hr_df.sort_values('HR', ascending=False).head(5)
                        st.markdown("#### Top covariables por efecto en abandono (HR)")
                        
                        for idx, (name, row) in enumerate(top.iterrows(), 1):
                            hr_val = float(row['HR'])
                            hr_pct = abs((hr_val - 1) * 100)
                            
                            # Determinar dirección del efecto
                            if hr_val > 1:
                                color = "#ff6b6b"
                                message = f"⚠️ **Aumenta el riesgo de abandono** en un {hr_pct:.2f}%"
                            else:
                                color = "#51cf66"
                                message = f"✅ **Reduce el riesgo de abandono** en un {hr_pct:.2f}%"
                            
                            # Mostrar con interpretación simple
                            st.markdown(f"**{idx}. {name}**")
                            st.markdown(f"<div style='background-color:{color}22; padding:10px; border-left:4px solid {color}; border-radius:5px;'>"
                                      f"{message}"
                                      f"</div>", unsafe_allow_html=True)
                        
                        # Conclusión general
                        st.markdown("---")
                        st.markdown("**Conclusión:**")
                        
                        accelerators = top[top['HR'] > 1]
                        retarders = top[top['HR'] < 1]
                        
                        conclusion = f"En el análisis de '{var}':\n"
                        
                        if len(accelerators) > 0:
                            acc_vars = ", ".join([f"**{n}**" for n in accelerators.index[:3]])
                            conclusion += f"\n🔴 **Variables que ACELERAN el abandono:** {acc_vars}\n"
                        
                        if len(retarders) > 0:
                            ret_vars = ", ".join([f"**{n}**" for n in retarders.index[:3]])
                            conclusion += f"\n🟢 **Variables que RETRASAN el abandono:** {ret_vars}\n"
                        
                        if len(accelerators) == 0 and len(retarders) == 0:
                            conclusion += "\nNo se encontraron variables con efecto significativo.\n"
                        
                        conclusion += f"\nEl modelo COX alcanza una precisión de predicción del {concord*100:.2f}% (C-index), indicando que es {'altamente' if concord > 0.7 else 'moderadamente' if concord > 0.6 else 'débilmente'} efectivo para predecir abandono en base a estos factores."
                        
                        st.info(conclusion)
                        
                    except Exception as e:
                        st.warning(f"No se pudo generar análisis detallado de top covariables: {e}")
                except Exception as e:
                    st.markdown(f"<h3 style='color:red'>No se pudo mostrar resultados COX</h3>\n\n**Razón:** {e}", unsafe_allow_html=True)

            def render_logrank():
                # Log-Rank: comparaciones por pares
                try:
                    st.write("**Log‑Rank (comparaciones por pares)**")
                    
                    # Curva KM (vista gráfica)
                    km_results = compute_km(df, 'Tenure Months', 'Churn Value', var)
                    if not km_results:
                        st.markdown("<h3 style='color:red'>No se pudo ajustar Kaplan–Meier: datos insuficientes</h3>", unsafe_allow_html=True)
                    else:
                        # Gráfico
                        if PLOTLY_AVAILABLE:
                            fig = go.Figure()
                            for grp, res in km_results.items():
                                if res is None:
                                    continue
                                fig.add_trace(go.Scatter(x=res['timeline'], y=res['survival'], mode='lines', 
                                                       name=f"Grupo: {grp}",
                                                       hovertemplate=f'Grupo: {grp}<br>Mes: %{{x}}<br>Supervivencia: %{{y:.3f}}<extra></extra>'))
                            fig.update_layout(title=f'Curva KM - {var}', xaxis_title='Tiempo (meses)', yaxis_title='Probabilidad de supervivencia', template='plotly_white')
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig, ax = plt.subplots(figsize=(10,6))
                            for grp, res in km_results.items():
                                if res is None:
                                    continue
                                ax.step(res['timeline'], res['survival'], where='post', label=f"Grupo: {grp}")
                            ax.set_xlabel('Tiempo (meses)')
                            ax.set_ylabel('Probabilidad de supervivencia')
                            ax.set_title(f'Curva KM - {var}')
                            ax.legend()
                            st.pyplot(fig)
                            plt.close(fig)

                    # Tabla con comparaciones por pares (Log-Rank)
                    pvals = compute_logrank_pairs(df, 'Tenure Months', 'Churn Value', var)
                    if not pvals:
                        st.write("No se pudo calcular Log-Rank")
                    else:
                        rows = []
                        p_list = []
                        for comp, vals in pvals.items():
                            p = vals.get('p_value')
                            hr = vals.get('HR')
                            if hr is not None:
                                hr_pct = abs((hr - 1) * 100)
                                # Extraer los grupos de la comparación
                                groups = comp.split(' vs ')
                                if hr > 1:
                                    effect = f"{groups[1]} tiene {hr_pct:.2f}% más riesgo que {groups[0]}"
                                else:
                                    effect = f"{groups[1]} tiene {hr_pct:.2f}% menos riesgo que {groups[0]}"
                            else:
                                effect = "N/A"
                            rows.append((comp, effect))
                            if p is not None:
                                p_list.append((comp, p, hr))

                        p_df = pd.DataFrame(rows, columns=['Comparación','Interpretación'])
                        st.dataframe(p_df)

                        # Conclusión
                        sig_pairs = [t for t in p_list if t[1] is not None and t[1] < 0.05]
                        if sig_pairs:
                            best = sorted(sig_pairs, key=lambda x: x[1])[0]
                            comp_best, p_best, hr_best = best
                            groups = comp_best.split(' vs ')
                            if hr_best is not None:
                                hr_pct = abs((hr_best - 1) * 100)
                                if hr_best > 1:
                                    hr_text = f"{groups[1]} tiene {hr_pct:.2f}% más riesgo de abandono que {groups[0]}"
                                else:
                                    hr_text = f"{groups[1]} tiene {hr_pct:.2f}% menos riesgo de abandono que {groups[0]}"
                            else:
                                hr_text = "N/A"
                            st.success(f"✓ Se detectan diferencias significativas en {var}.\n\nComparación más relevante: {hr_text}")
                        else:
                            st.info(f"ℹ No se detectan diferencias significativas entre los grupos de {var}.")
                except Exception as e:
                    st.markdown(f"<h3 style='color:red'>No se pudo calcular Log‑Rank</h3>\n\n**Razón:** {e}", unsafe_allow_html=True)

            if view_choice == "Log‑Rank":
                render_logrank()
            elif view_choice == "Cox":
                render_cox()

            st.markdown('---')

st.markdown('')
# Sugerencias contextuales según el modelo seleccionado
if view_choice == "KM":
    st.write('Sugerencia: Log-Rank te muestra comparaciones por pares para cada variable seleccionada.')
elif view_choice == "Log‑Rank":
    st.write('Sugerencia: COX te mostrará qué variables impulsarán el riesgo instantáneo de abandono.')

st.markdown(
    "<div style='margin-top: 8rem; padding: 0.8rem 1rem; text-align: center; "
    "border-top: 1px solid #ddd; color: #ffffff; font-size: 1rem; font-weight: 600;'>"
    "✨ Creado por Juan Carlos Monte de Oca✨"
    "</div>",
    unsafe_allow_html=True
)
