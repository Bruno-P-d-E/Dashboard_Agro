import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import re
from scipy import stats
import pydeck as pdk


# Configuração da página
st.set_page_config(
    page_title="Dashboard Soja Paraná",
    page_icon="🌱",
    layout="wide"
)

# CSS customizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #2c5f2d;
        text-align: center;
        padding: 20px;
        margin-bottom: 10px;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Título
st.markdown("<h1>🌱 Dashboard - Soja no Paraná (2018-2024)</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #666;'>Análise Inteligente: Clima + Produtividade + Geolocalização</h3>", unsafe_allow_html=True)

# Carregar dados
@st.cache_data
def carregar_dados():
    try:
        df = pd.read_csv('PAM_SIDRA_NASAPOWER_FENOLOGIA_SOJA_PR_Copia.csv')
        
        # Calcular área perdida
        df['Área perdida (Hectares)'] = df['Área plantada (Hectares)'] - df['Área colhida (Hectares)']
        df['Percentual de perda (%)'] = (df['Área perdida (Hectares)'] / df['Área plantada (Hectares)']) * 100
        
        # Renomear coluna para merge - NOVA MUDANÇA
        df = df.rename(columns={'Código IBGE': 'codigo_ibge'})
        df['codigo_ibge'] = df['codigo_ibge'].astype(str).str.zfill(7).str[:7].astype(int) # Garantir que o IBGE tenha 7 dígitos
        
        return df
    except FileNotFoundError:
        st.error("⚠️ Erro: Arquivo 'PAM_SIDRA_NASAPOWER_FENOLOGIA_SOJA_PR_Copia.csv' não encontrado!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        st.stop()

@st.cache_data
def carregar_municipios():
    try:
        df_municipios = pd.read_csv('municipios.csv')
        df_parana = df_municipios[df_municipios['codigo_uf'] == 41].copy()
        df_parana = df_parana.rename(columns={'longitude': 'lon', 'latitude': 'lat'})
        df_parana['codigo_ibge'] = df_parana['codigo_ibge'].astype(str).str.zfill(7).str[:7].astype(int) # Garantir que o IBGE tenha 7 dígitos para merge
        return df_parana
    except FileNotFoundError:
        st.warning("⚠️ Arquivo 'municipios.csv' não encontrado. Mapa 3D não disponível.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Erro ao carregar municípios: {e}")
        return None

df = carregar_dados()
df_municipios = carregar_municipios()

# Identificar colunas climáticas
colunas_climaticas = [col for col in df.columns if re.match(r'.*_dec\d+_ano\d+', col)]
atributos_climaticos = list(set([col.rsplit('_dec', 1)[0] for col in colunas_climaticas]))

# Função para calcular correlações com variáveis de soja
@st.cache_data
def calcular_correlacoes_relevantes(_df):
    variaveis_soja = [
        'Rendimento médio da produção (Quilogramas por Hectare)',
        'Quantidade produzida (Toneladas)',
        'Área perdida (Hectares)',
        'Percentual de perda (%)'
    ]
    
    resultados = []
    
    for var_soja in variaveis_soja:
        for col_clima in colunas_climaticas:
            try:
                df_temp = _df[[col_clima, var_soja]].dropna()
                if len(df_temp) > 0:
                    corr = df_temp.corr().iloc[0, 1]
                    if not np.isnan(corr):
                        atributo = col_clima.rsplit('_dec', 1)[0]
                        dec_match = re.search(r'dec(\d+)', col_clima)
                        ano_match = re.search(r'ano(\d+)', col_clima)
                        
                        if dec_match and ano_match:
                            resultados.append({
                                'Variável Climática': atributo,
                                'Decêndio': int(dec_match.group(1)),
                                'Ano Safra': f"ano{ano_match.group(1)}",
                                'Coluna': col_clima,
                                'Variável Soja': var_soja,
                                'Correlação': corr,
                                'Correlação Abs': abs(corr)
                            })
            except:
                continue
    
    return pd.DataFrame(resultados)

with st.spinner("🔍 Analisando correlações climáticas..."):
    df_correlacoes_inicial = calcular_correlacoes_relevantes(df)

# Sidebar - Filtros
st.sidebar.header("🔍 Filtros de Análise")

# Filtro de Anos
anos_disponiveis = sorted(df['ano'].unique())
anos_selecionados = st.sidebar.multiselect(
    "Selecione os anos:",
    options=anos_disponiveis,
    default=anos_disponiveis
)

# Filtro de Municípios
municipios_disponiveis = sorted(df['Município'].unique())
visualizar_todos = st.sidebar.radio(
    "Municípios:",
    options=["Todos os municípios", "Selecionar específicos"],
    index=0
)

if visualizar_todos == "Todos os municípios":
    municipios_selecionados = municipios_disponiveis
else:
    municipios_selecionados = st.sidebar.multiselect(
        "Escolha os municípios:",
        options=municipios_disponiveis,
        default=municipios_disponiveis[:5] if len(municipios_disponiveis) >= 5 else municipios_disponiveis
    )

# Aplicar filtros
df_filtrado = df[
    (df['ano'].isin(anos_selecionados)) & 
    (df['Município'].isin(municipios_selecionados))
].copy()

# Informações
st.sidebar.markdown("---")
st.sidebar.header("📊 Informações")
st.sidebar.metric("Municípios", len(municipios_selecionados))
st.sidebar.metric("Anos", len(anos_selecionados))
st.sidebar.metric("Registros", len(df_filtrado))
st.sidebar.metric("Variáveis Climáticas", len(colunas_climaticas))

# Agregação por ano
df_agregado = df_filtrado.groupby('ano').agg({
    'Área plantada (Hectares)': 'sum',
    'Área colhida (Hectares)': 'sum',
    'Área perdida (Hectares)': 'sum',
    'Percentual de perda (%)': 'mean',
    'Quantidade produzida (Toneladas)': 'sum',
    'Valor da produção (Mil Reais)': 'sum',
    'Rendimento médio da produção (Quilogramas por Hectare)': 'mean'
}).reset_index()

# ===========================
# MÉTRICAS PRINCIPAIS
# ===========================
st.header("📊 Indicadores Principais – Paraná (Último Ano)")
st.info("📋 Resumo dos principais indicadores de produção e rendimento de soja no último ano agrícola.")

if len(df_agregado) > 0:
    ultimo_ano = df_agregado.iloc[-1]
    penultimo_ano = df_agregado.iloc[-2] if len(df_agregado) > 1 else ultimo_ano
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        area_var = ((ultimo_ano['Área plantada (Hectares)'] - penultimo_ano['Área plantada (Hectares)']) / penultimo_ano['Área plantada (Hectares)'] * 100) if penultimo_ano['Área plantada (Hectares)'] > 0 else 0
        st.metric("Área Plantada", f"{ultimo_ano['Área plantada (Hectares)']:,.0f} ha", f"{area_var:+.2f}%")
    
    with col2:
        perda_var = ((ultimo_ano['Área perdida (Hectares)'] - penultimo_ano['Área perdida (Hectares)']) / penultimo_ano['Área perdida (Hectares)'] * 100) if penultimo_ano['Área perdida (Hectares)'] > 0 else 0
        st.metric("Área Perdida", f"{ultimo_ano['Área perdida (Hectares)']:,.0f} ha", f"{perda_var:+.2f}%", delta_color="inverse")
    
    with col3:
        prod_var = ((ultimo_ano['Quantidade produzida (Toneladas)'] - penultimo_ano['Quantidade produzida (Toneladas)']) / penultimo_ano['Quantidade produzida (Toneladas)'] * 100) if penultimo_ano['Quantidade produzida (Toneladas)'] > 0 else 0
        st.metric("Produção", f"{ultimo_ano['Quantidade produzida (Toneladas)']:,.0f} t", f"{prod_var:+.2f}%")
    
    with col4:
        rend_var = ((ultimo_ano['Rendimento médio da produção (Quilogramas por Hectare)'] - penultimo_ano['Rendimento médio da produção (Quilogramas por Hectare)']) / penultimo_ano['Rendimento médio da produção (Quilogramas por Hectare)'] * 100) if penultimo_ano['Rendimento médio da produção (Quilogramas por Hectare)'] > 0 else 0
        st.metric("Rendimento", f"{ultimo_ano['Rendimento médio da produção (Quilogramas por Hectare)']:,.0f} kg/ha", f"{rend_var:+.2f}%")
    
    with col5:
        st.metric("% de Perda", f"{ultimo_ano['Percentual de perda (%)']:.2f}%", 
                     f"{(ultimo_ano['Percentual de perda (%)'] - penultimo_ano['Percentual de perda (%)']):+.2f}pp", 
                     delta_color="inverse")

# ===========================
# MAPA 3D INTERATIVO - CORREÇÃO FINAL
# ===========================
if df_municipios is not None:
    st.header("🗺️ Mapa 3D – Distribuição Espacial da Produção")
    st.info("📋 Visualização tridimensional da variáveis de produção de soja por município. A altura das colunas representa o volume")
    
    # Seleção de ano para o mapa
    anos_mapa_disponiveis = sorted(df_filtrado['ano'].unique())
    if len(anos_mapa_disponiveis) > 0:
        ano_mapa = st.selectbox("Selecione o ano para visualização:", anos_mapa_disponiveis, 
                                index=len(anos_mapa_disponiveis)-1, key='ano_mapa')
    else:
        st.warning("Não há anos disponíveis nos filtros para o mapa.")
        ano_mapa = None

    if ano_mapa is not None:
        # Preparar dados para o mapa
        df_ano_mapa = df_filtrado[df_filtrado['ano'] == ano_mapa].copy()
        
        # Fazer o merge usando codigo_ibge
        df_mapa = df_municipios.merge(
            df_ano_mapa,
            on='codigo_ibge',
            how='inner'
        )
        
        if len(df_mapa) > 0:
            # Seleção de métrica para visualização
            col1, col2, col3 = st.columns(3)
            
            with col1:
                metrica_mapa = st.selectbox(
                    "Métrica para visualização:",
                    ["Quantidade produzida (Toneladas)", 
                     "Rendimento médio da produção (Quilogramas por Hectare)",
                     "Área perdida (Hectares)",
                     "Percentual de perda (%)",
                     "Valor da produção (Mil Reais)"],
                    key='metrica_mapa'
                )
            
            with col2:
                column_width = st.slider("Largura das Colunas (metros)", 3000, 30000, 15000, 1000)
            
            with col3:
                elevation_scale = st.slider("Escala de Elevação", 5, 50, 20, 5)
            
            # Controle de altura máxima
            elevation_max = st.slider("Altura Máxima", 5000, 20000, 10000, 1000)
            
            # Preparar dados para PyDeck
            df_mapa['metrica_viz'] = df_mapa[metrica_mapa]
            
            # Normalizar para cor e elevação
            max_metrica = df_mapa['metrica_viz'].max()
            df_mapa['elevation'] = (df_mapa['metrica_viz'] / max_metrica) * elevation_max
            
            # Definição do Color Range (mantida)
            COLOR_RANGE = [
                [255, 255, 178], [254, 204, 92], [253, 141, 60],
                [240, 59, 32], [189, 0, 38], [128, 0, 38]
            ]
            
            # Função auxiliar para mapeamento de cor (ajustada a partir da correção anterior)
            def map_value_to_color(value, data_max, color_range, alpha=200):
                """Mapeia um valor numérico para uma cor no COLOR_RANGE."""
                if data_max == 0 or np.isnan(value) or value == 0:
                    return [150, 150, 150, alpha] # Cor cinza para NaN ou zero
                
                # Para evitar erro de logarítmico (para escala de cor)
                # Normaliza o valor para uma escala de 0 a 1
                normalized_value = (value - df_mapa['metrica_viz'].min()) / (data_max - df_mapa['metrica_viz'].min())
                
                num_colors = len(color_range)
                # Garante que o índice fique dentro dos limites [0, num_colors-1]
                index = int(normalized_value * (num_colors - 1))
                
                return color_range[index] + [alpha]

            # Criar a coluna de cor final no DataFrame usando o mapeamento
            df_mapa['fill_color'] = df_mapa['metrica_viz'].apply(
                lambda x: map_value_to_color(x, max_metrica, COLOR_RANGE)
            )

            # Criar camada de Colunas (ColumnLayer)
            column_layer = pdk.Layer(
                "ColumnLayer",
                data=df_mapa,
                get_position="[lon, lat]",
                get_elevation="elevation",
                elevation_scale=elevation_scale,
                radius=column_width,
                # CORREÇÃO: Referencia a nova coluna 'fill_color'
                get_fill_color="fill_color", 
                # Adiciona as colunas para o tooltip reconhecer no hover
                # A coluna 'nome' já está disponível; 'metrica_viz' é a coluna com o valor
                get_tooltip=['nome', 'metrica_viz'], 
                pickable=True,
                auto_highlight=True,
                extruded=True,
            )
            
            # Centro do mapa
            center_lat = df_mapa['lat'].mean()
            center_lon = df_mapa['lon'].mean()
            
            # Criar visualização
            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=6.5,
                pitch=50,
                bearing=0
            )
            
            # Tooltip ajustado para sintaxe Deck.gl e nome da métrica
            # Removido o formato numérico complexo para evitar erro de parse
            metrica_nome_tooltip = metrica_mapa.split('(')[0].strip()
            tooltip = {
                "html": f"<b>Município:</b> {{nome}}<br/>"
                        f"<b>{metrica_nome_tooltip}:</b> {{metrica_viz}}",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }
            
            # Renderizar mapa
            st.pydeck_chart(pdk.Deck(
                layers=[column_layer],
                initial_view_state=view_state,
                tooltip=tooltip
            ))
            
            # ===========================
            # NOVO: Legenda de Cores (Mapa Abaixo)
            # ===========================
            st.subheader("🎨 Legenda de Cores")
            
            # Calcula os limites para cada cor
            min_val = df_mapa['metrica_viz'].min()
            max_val = df_mapa['metrica_viz'].max()
            step = (max_val - min_val) / len(COLOR_RANGE)
            
            legend_html = f"<b>{metrica_nome_tooltip}:</b>"
            legend_html += "<div style='display: flex; flex-direction: column;'>"
            
            for i, color_rgb in enumerate(COLOR_RANGE):
                # O Pydeck usa a cor mais escura para o valor mais alto.
                color_index = len(COLOR_RANGE) - 1 - i
                color = COLOR_RANGE[color_index]
                
                # Calcula o limite inferior e superior para o texto da legenda
                lower_bound = min_val + (color_index * step)
                upper_bound = min_val + ((color_index + 1) * step)
                
                # Cor no formato CSS
                css_color = f"rgb({color[0]}, {color[1]}, {color[2]})"
                
                legend_html += f"<div style='display: flex; align-items: center; margin-bottom: 3px;'><br/><div style='width: 20px; height: 10px; background-color: {css_color}; margin-right: 10px; border: 1px solid #333;'></div><br/><span>{upper_bound:.2f} (Máx)</span></div>"
            
            # Adiciona o limite mínimo e reverte a ordem para a legenda
            legend_html = legend_html.replace(f"{max_val:.2f} (Máx)", f"{max_val:.2f} (Máx)")
            legend_html += f"""
                <div style='margin-top: 5px; text-align: left;'>
                    <span>{min_val:.2f} (Min)</span>
                </div>
            </div>"""
            
            st.markdown(legend_html, unsafe_allow_html=True)
            
            # Estatísticas do mapa (mantidas)
            col1, col2, col3, col4 = st.columns(4)
            # ... (continuação das estatísticas) ...
            with col1:
                st.metric("Municípios no Mapa", len(df_mapa))
            with col2:
                st.metric(f"Média - {metrica_mapa.split('(')[0].strip()}", f"{df_mapa['metrica_viz'].mean():.2f}")
            with col3:
                st.metric("Máximo", f"{df_mapa['metrica_viz'].max():.2f}")
            with col4:
                st.metric("Mínimo", f"{df_mapa['metrica_viz'].min():.2f}")
            
            # Top 10 municípios no mapa (mantidos)
            with st.expander("🏆 Top 10 Municípios - Visualização Detalhada"):
                top_10_mapa = df_mapa.nlargest(10, 'metrica_viz')[['nome', metrica_mapa]]
                st.dataframe(top_10_mapa, hide_index=True, use_container_width=True)
        else:
            st.warning("⚠️ Não foi possível fazer o merge dos dados geográficos (código IBGE) para o ano selecionado. Verifique a coluna 'Código IBGE' no arquivo PAM.")
    else:
        st.info("ℹ️ Selecione um ano disponível nos filtros para exibir o mapa.")
else:
    st.info("ℹ️ Mapa 3D não disponível - arquivo 'municipios.csv' não encontrado.")

    
# ===========================
# GRÁFICOS PRINCIPAIS
# ===========================
st.header("📈Análise Produtiva")
st.info("📈 Avaliação temporal da evolução da área cultivada, perdas percentuais e variação da produtividade.")

col1, col2 = st.columns(2)

with col1:
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_agregado['ano'], y=df_agregado['Área plantada (Hectares)'],
                              name='Plantada', line=dict(color='#2ecc71', width=3), mode='lines+markers'))
    fig1.add_trace(go.Scatter(x=df_agregado['ano'], y=df_agregado['Área colhida (Hectares)'],
                              name='Colhida', line=dict(color='#27ae60', width=3), mode='lines+markers'))
    fig1.add_trace(go.Scatter(x=df_agregado['ano'], y=df_agregado['Área perdida (Hectares)'],
                              name='Perdida', line=dict(color='#e74c3c', width=3), fill='tozeroy', mode='lines+markers'))
    fig1.update_layout(title='<b>Evolução da Área e Perdas: mostra o comportamento da área plantada versus área perdida ao longo dos anos</b>', 
                      xaxis_title='Ano', yaxis_title='Hectares', hovermode='x unified', height=450)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=df_agregado['ano'], y=df_agregado['Quantidade produzida (Toneladas)'],
                          name='Produção', marker_color='#3498db'), secondary_y=False)
    fig2.add_trace(go.Scatter(x=df_agregado['ano'], y=df_agregado['Percentual de perda (%)'],
                              name='% Perda', line=dict(color='#e74c3c', width=3), mode='lines+markers'), secondary_y=True)
    fig2.update_layout(title='<b>Produtividade Média: acompanha o rendimento médio por hectare ao longo dos anos</b>', hovermode='x unified', height=450)
    fig2.update_xaxes(title_text="Ano")
    fig2.update_yaxes(title_text="Toneladas", secondary_y=False)
    fig2.update_yaxes(title_text="% Perda", secondary_y=True)
    st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df_agregado['ano'], y=df_agregado['Rendimento médio da produção (Quilogramas por Hectare)'],
                              mode='lines+markers', line=dict(color='#9b59b6', width=3), marker=dict(size=12)))
    fig3.update_layout(title='<b>Rendimento Médio: representa o rendimento médio da produção quilogramas por hectare ao longo dos anos</b>', xaxis_title='Ano', yaxis_title='kg/ha', height=400)
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=df_agregado['ano'], y=df_agregado['Valor da produção (Mil Reais)']/1000,
                          marker_color='#16a085', text=df_agregado['Valor da produção (Mil Reais)']/1000,
                          texttemplate='R$ %{text:.1f}M', textposition='outside'))
    fig4.update_layout(title='<b>Valor da Produção: evolução do valor econômico total (R$) ao longo dos anos</b>', xaxis_title='Ano', yaxis_title='Milhões R$', height=400)
    st.plotly_chart(fig4, use_container_width=True)

# ===========================
# VARIÁVEIS CLIMÁTICAS MAIS RELEVANTES
# ===========================
st.header("🌤️ Variáveis Climáticas Mais Relevantes")

st.info("📋 **Análise automática:** Identificando as variáveis climáticas com maior correlação com rendimento, produção e perdas.")

# Filtros principais
col1, col2, col3 = st.columns(3)

with col1:
    top_n = st.slider("Número de variáveis mais relevantes:", 5, 20, 10)

with col2:
    metrica_foco = st.selectbox("Foco da análise:", [
        'Rendimento médio da produção (Quilogramas por Hectare)',
        'Quantidade produzida (Toneladas)',
        'Área perdida (Hectares)',
        'Percentual de perda (%)'
    ])

with col3:
    ano_clima_analise = st.selectbox(
        "Ano para análise:",
        options=["Todos os anos"] + [str(ano) for ano in sorted(anos_selecionados)],
        index=0
    )

# Filtrar dados por ano se necessário
if ano_clima_analise == "Todos os anos":
    df_para_correlacao = df_filtrado.copy()
    titulo_ano = "Todos os Anos"
else:
    df_para_correlacao = df_filtrado[df_filtrado['ano'] == int(ano_clima_analise)].copy()
    titulo_ano = ano_clima_analise

# Recalcular correlações com o filtro de ano
@st.cache_data
def calcular_correlacoes_por_ano(_df, metrica, ano_filtro):
    resultados = []
    
    for col_clima in colunas_climaticas:
        try:
            df_temp = _df[[col_clima, metrica]].dropna()
            if len(df_temp) > 5:
                corr = df_temp.corr().iloc[0, 1]
                if not np.isnan(corr):
                    atributo = col_clima.rsplit('_dec', 1)[0]
                    dec_match = re.search(r'dec(\d+)', col_clima)
                    ano_match = re.search(r'ano(\d+)', col_clima)
                    
                    if dec_match and ano_match:
                        resultados.append({
                            'Variável Climática': atributo,
                            'Decêndio': int(dec_match.group(1)),
                            'Ano Safra': f"ano{ano_match.group(1)}",
                            'Coluna': col_clima,
                            'Correlação': corr,
                            'Correlação Abs': abs(corr)
                        })
        except:
            continue
    
    return pd.DataFrame(resultados)

df_corr_foco = calcular_correlacoes_por_ano(df_para_correlacao, metrica_foco, ano_clima_analise)

if len(df_corr_foco) == 0:
    st.warning("⚠️ Não há dados suficientes para calcular correlações com os filtros selecionados.")
    st.stop()

df_corr_foco = df_corr_foco.nlargest(min(top_n, len(df_corr_foco)), 'Correlação Abs')

# Gráfico de barras das correlações mais fortes
st.subheader(f"🔝 Top {len(df_corr_foco)} Variáveis com Maior Impacto - {titulo_ano}")

fig_top = go.Figure()
fig_top.add_trace(go.Bar(
    x=df_corr_foco['Correlação'],
    y=[f"{row['Variável Climática']}_dec{row['Decêndio']}_{row['Ano Safra']}" for _, row in df_corr_foco.iterrows()],
    orientation='h',
    marker_color=df_corr_foco['Correlação'],
    marker_colorscale='RdYlGn',
    marker_cmin=-1,
    marker_cmax=1,
    text=df_corr_foco['Correlação'],
    texttemplate='%{text:.3f}',
    textposition='outside'
))

fig_top.update_layout(
    title=f'<b>Correlação com: {metrica_foco} ({titulo_ano})</b>',
    xaxis_title='Correlação de Pearson',
    yaxis_title='Variável Climática',
    height=max(400, len(df_corr_foco) * 30),
    xaxis_range=[-1, 1]
)
fig_top.add_vline(x=0, line_dash="dash", line_color="gray")
st.plotly_chart(fig_top, use_container_width=True)

# Análise detalhada das top 3
st.subheader("🔍 Análise Detalhada – Top 3 Variáveis (Todos os Anos)")
st.info(f"🔬 Relação entre as três variáveis climáticas de maior impacto e a produtividade média da soja, com classificação automática da força e direção da correlação - {titulo_ano}")

n_pontos = len(df_para_correlacao)
st.info(f"📊 Análise baseada em **{n_pontos} registros** ({titulo_ano})")

top3 = df_corr_foco.head(3)

for idx, row in top3.iterrows():
    with st.expander(f"**{idx+1}. {row['Variável Climática']} - Decêndio {row['Decêndio']} ({row['Ano Safra']})** - Correlação: {row['Correlação']:.4f}"):
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            df_scatter = df_para_correlacao[[row['Coluna'], metrica_foco, 'ano', 'Município', 'Quantidade produzida (Toneladas)']].dropna()
            
            try:
                fig_scatter = px.scatter(
                    df_scatter,
                    x=row['Coluna'],
                    y=metrica_foco,
                    color='ano',
                    size='Quantidade produzida (Toneladas)',
                    hover_data=['Município'],
                    trendline='ols',
                    title=f"Mostram o comportamento de dispersão ({metrica_foco.split('(')[0].strip()}) × ({row['Variável Climática']})"
                )
            except:
                fig_scatter = px.scatter(
                    df_scatter,
                    x=row['Coluna'],
                    y=metrica_foco,
                    color='ano',
                    size='Quantidade produzida (Toneladas)',
                    hover_data=['Município'],
                    title=f"Mostram o comportamento de dispersão ({metrica_foco.split('(')[0].strip()}) × ({row['Variável Climática']})"
                )
                
                if len(df_scatter) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(df_scatter[row['Coluna']], df_scatter[metrica_foco])
                    line_x = np.array([df_scatter[row['Coluna']].min(), df_scatter[row['Coluna']].max()])
                    line_y = slope * line_x + intercept
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode='lines',
                        name='Tendência',
                        line=dict(color='red', dash='dash', width=2)
                    ))
            
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            st.metric("Correlação", f"{row['Correlação']:.4f}")
            
            if abs(row['Correlação']) > 0.7:
                intensidade = "🔴 Forte"
            elif abs(row['Correlação']) > 0.4:
                intensidade = "🟡 Moderada"
            else:
                intensidade = "🟢 Fraca"
            
            st.metric("Indica intensidade (🟢 Fraca, 🟡 Moderada, 🔴 Forte)", intensidade)
            
            direcao = "📈 Positiva" if row['Correlação'] > 0 else "📉 Negativa"
            st.metric("direção (positiva ou negativa)", direcao)
            
            st.markdown("**Caixa de interpretação rápida:** explica o tipo de associação observada.")
            if row['Correlação'] > 0:
                st.success(f"Aumento de {row['Variável Climática']} pode estar associado ao aumento de {metrica_foco.split('(')[0].strip()}")
            else:
                st.warning(f"Aumento de {row['Variável Climática']} pode estar associado à redução de {metrica_foco.split('(')[0].strip()}")

# ===========================
# MAPA DE CALOR: Correlações por Decêndio
# ===========================
st.header(f"🗺️ Mapa de Calor: Ciclo Completo da Safra - {titulo_ano}")
# descrição do mapa de calor

st.info("📅 **Ciclo da Soja:** Ano 1 (Dec 26-36: Set-Dez) → Ano 2 (Dec 1-15: Jan-Mai)" \
''' Mapa de correlações entre variáveis climáticas e produtividade ao longo das fases fenológicas da cultura.
Permite identificar os períodos de maior sensibilidade climática e variáveis críticas por estágio.''')


variaveis_disponiveis = sorted(df_corr_foco['Variável Climática'].unique())
vars_heatmap = st.multiselect(
    "Selecione variáveis climáticas para o mapa de calor:",
    options=variaveis_disponiveis,
    default=variaveis_disponiveis[:min(5, len(variaveis_disponiveis))]
)

if vars_heatmap:
    decendios_ano1 = list(range(26, 37))
    decendios_ano2 = list(range(1, 16))
    
    heatmap_data = []
    
    for var_clima in vars_heatmap:
        # Ano 1: decêndios 26-36
        for dec_num in decendios_ano1:
            col_name = f"{var_clima}_dec{dec_num}_ano1"
            if col_name in df_para_correlacao.columns:
                try:
                    df_temp = df_para_correlacao[[col_name, metrica_foco]].dropna()
                    if len(df_temp) > 5:
                        corr = df_temp.corr().iloc[0, 1]
                        if not np.isnan(corr):
                            heatmap_data.append({
                                'Variável': var_clima,
                                'Período': f"Ano1_Dec{dec_num}",
                                'Decêndio_Order': dec_num - 26,
                                'Correlação': corr
                            })
                except:
                    pass
        
        # Ano 2: decêndios 1-15
        for dec_num in decendios_ano2:
            col_name = f"{var_clima}_dec{dec_num}_ano2"
            if col_name in df_para_correlacao.columns:
                try:
                    df_temp = df_para_correlacao[[col_name, metrica_foco]].dropna()
                    if len(df_temp) > 5:
                        corr = df_temp.corr().iloc[0, 1]
                        if not np.isnan(corr):
                            heatmap_data.append({
                                'Variável': var_clima,
                                'Período': f"Ano2_Dec{dec_num}",
                                'Decêndio_Order': 11 + (dec_num - 1),
                                'Correlação': corr
                            })
                except:
                    pass
    
    df_heatmap = pd.DataFrame(heatmap_data)
    
    if len(df_heatmap) > 0:
        pivot_heatmap = df_heatmap.pivot_table(
            values='Correlação',
            index='Variável',
            columns='Período',
            aggfunc='first'
        )
        
        colunas_ordenadas = [f"Ano1_Dec{i}" for i in decendios_ano1] + [f"Ano2_Dec{i}" for i in decendios_ano2]
        colunas_presentes = [col for col in colunas_ordenadas if col in pivot_heatmap.columns]
        pivot_heatmap = pivot_heatmap[colunas_presentes]
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=pivot_heatmap.values,
            x=pivot_heatmap.columns,
            y=pivot_heatmap.index,
            colorscale='RdYlGn',
            zmid=0,
            text=pivot_heatmap.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 8},
            colorbar=dict(title="Correlação"),
            zmin=-1,
            zmax=1
        ))
        
        fig_heatmap.update_layout(
            title=f'<b>Correlação ao longo do Ciclo da Safra: {metrica_foco.split("(")[0].strip()} ({titulo_ano})</b>',
            xaxis_title='Período (Ano1: Set-Dez | Ano2: Jan-Mai)',
            yaxis_title='Variável Climática',
            height=max(500, len(pivot_heatmap) * 70),
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=9)
            )
        )
        
        fig_heatmap.add_vline(x=10.5, line_dash="dash", line_color="white", line_width=2)
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Resumo por fase
        st.subheader("📊 Correlação Média por Fase da Safra")
        st.info("📋 Correlação média consolidada das variáveis climáticas mais relevantes em cada fase fenológica, indicando os estágios com maior dependência climática da produtividade.")
        col1, col2 = st.columns(2)
        
        with col1:
            ano1_cols = [col for col in pivot_heatmap.columns if col.startswith("Ano1")]
            if ano1_cols:
                ano1_data = pivot_heatmap[ano1_cols]
                st.metric("Fase 1: Plantio/Desenvolvimento", 
                          f"{ano1_data.mean().mean():.4f}",
                          help="Ano1 Dec26-36: Set-Dez")
        
        with col2:
            ano2_cols = [col for col in pivot_heatmap.columns if col.startswith("Ano2")]
            if ano2_cols:
                ano2_data = pivot_heatmap[ano2_cols]
                st.metric("Fase 2: Floração/Maturação", 
                          f"{ano2_data.mean().mean():.4f}",
                          help="Ano2 Dec1-15: Jan-Mai")

# ===========================
# RANKING DE MUNICÍPIOS
# ===========================
st.header("🏘️ Ranking de Municípios")
st.info("📋 Classificação dos municípios paranaenses com melhor desempenho produtivo e econômico na soja, considerando área cultivada, produtividade média e valor total da produção.")
ano_rank = st.selectbox("Ano para ranking:", anos_selecionados, index=len(anos_selecionados)-1)
df_ano = df_filtrado[df_filtrado['ano'] == ano_rank].copy()

col1, col2, col3 = st.columns(3)

with col1:
    top_prod = df_ano.nlargest(10, 'Quantidade produzida (Toneladas)')
    fig_p = px.bar(top_prod, x='Quantidade produzida (Toneladas)', y='Município', orientation='h',
                   title=f'<b>Top 10 – Produção Total (toneladas)({ano_rank})</b>', color='Quantidade produzida (Toneladas)',
                   color_continuous_scale='Greens')
    fig_p.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_p, use_container_width=True)

with col2:
    top_rend = df_ano.nlargest(10, 'Rendimento médio da produção (Quilogramas por Hectare)')
    fig_r = px.bar(top_rend, x='Rendimento médio da produção (Quilogramas por Hectare)', y='Município',
                   orientation='h', title=f'<b>Top 10 – Produtividade Média (kg/ha) ({ano_rank})</b>',
                   color='Rendimento médio da produção (Quilogramas por Hectare)', color_continuous_scale='Blues')
    fig_r.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_r, use_container_width=True)

with col3:
    top_perda = df_ano.nlargest(10, 'Área perdida (Hectares)')
    fig_pp = px.bar(top_perda, x='Área perdida (Hectares)', y='Município', orientation='h',
                    title=f'<b>Top 10 – Área Plantada (ha) ({ano_rank})</b>', color='Área perdida (Hectares)',
                    color_continuous_scale='Reds')
    fig_pp.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_pp, use_container_width=True)

# Rodapé
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #683;'>
        🌱 <b>Dashboard Inteligente - Soja Paraná</b> | 
        Fonte: PAM/SIDRA + NASA POWER | Desenvolvido por: Bruno Proença
    </div>
""", unsafe_allow_html=True)