import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from motor_mdo import load_selig_from_lines, run_optimization

st.set_page_config(page_title="Otimizador MDO Aerodesign", layout="wide")
st.title("Otimizador Paramétrico de Aerofólios (Bézier + XFOIL Multi-Re)")
st.markdown("Carregue um perfil baseline (`.dat`), insira o envelope de Reynolds e as restrições estruturais/estabilidade do **MDO**.")

with st.sidebar:
    st.header("1. Perfil Baseline")
    uploaded_file = st.file_uploader("Upload do arquivo .dat (Selig)", type=["dat", "txt"])

    st.header("2. Intervalo de Reynolds (MDO)")
    re_min = st.number_input("Re Mínimo (Ponta / Decolagem)", value=150000, step=10000)
    re_med = st.number_input("Re Médio (Seção Média)", value=250000, step=10000)
    re_max = st.number_input("Re Máximo (Raiz / Cruzeiro)", value=350000, step=10000)

    st.header("3. Restrições Geométricas e Estabilidade")
    min_max_tc = st.slider("Espessura Máx. Mínima (% corda)", 8.0, 18.0, 11.5, 0.5) / 100.0
    max_max_tc = st.slider("Espessura Máx. Limite (% corda)", 10.0, 22.0, 14.5, 0.5) / 100.0
    spar_x = st.slider("Posição da Longarina (% corda)", 15, 45, 30, 5) / 100.0
    min_spar_tc = st.slider("Espessura Mín. na Longarina (% corda)", 6.0, 16.0, 11.0, 0.5) / 100.0
    min_te_tc = st.number_input("Espessura Mín. Bordo de Fuga em 95%c", value=0.004, step=0.001, format="%.3f")
    min_cm = st.number_input("Limite de Momento Picador (Cm mín.)", value=-0.32, step=0.02)

    st.header("4. Objetivo da Missão")
    peso_cl = st.slider("Peso Sustentação Máxima (CL_max)", 0.0, 1.0, 0.65, 0.05)
    peso_ld = round(1.0 - peso_cl, 2)
    st.caption(f"Peso Eficiência Aerodinâmica (CL/CD): **{peso_ld:.2f}**")
    max_fev = st.slider("Máximo de Avaliações do Otimizador", 100, 600, 250, 50)

if uploaded_file is not None:
    lines = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    upper_orig, lower_orig = load_selig_from_lines(lines)

    if st.button("Iniciar Otimização Multi-Reynolds", type="primary"):
        re_list = [int(re_min), int(re_med), int(re_max)]
        mdo_specs = {
            "min_max_tc": min_max_tc,
            "max_max_tc": max_max_tc,
            "spar_x": spar_x,
            "min_spar_tc": min_spar_tc,
            "min_te_tc": min_te_tc,
            "min_cm": min_cm
        }
        weights = {"cl_max": peso_cl, "ld_max": peso_ld}

        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_ui(curr, total, best_sc):
            progress_bar.progress(min(curr / total, 1.0))
            status_text.text(f"Avaliação {curr}/{total} | Melhor Score MDO atual: {best_sc:.4f}")

        with st.spinner("Rodando simulações no XFOIL e ajustando pontos de Bézier..."):
            base_coords, opt_coords, base_sc, opt_sc, history = run_optimization(
                upper_orig, lower_orig, re_list, mdo_specs, weights, max_fev=max_fev, progress_callback=update_ui
            )

        progress_bar.progress(1.0)
        st.success(f"Otimização concluída! Score subiu de **{base_sc:.4f}** para **{opt_sc:.4f}**.")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(9, 3.8))
            ax.plot(base_coords[:, 0], base_coords[:, 1], "k--", linewidth=1.5, label="Baseline (Bézier)")
            ax.plot(opt_coords[:, 0], opt_coords[:, 1], "b-", linewidth=2.2, label="Otimizado MDO")
            ax.axvline(spar_x, color="orange", linestyle=":", label=f"Longarina ({spar_x*100:.0f}%c)")
            ax.set_title("Comparação Geométrica: Baseline vs. Otimizado")
            ax.set_xlabel("x / c")
            ax.set_ylabel("y / c")
            ax.axis("equal")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend()
            st.pyplot(fig)

        with col2:
            st.subheader("Exportar Resultado")
            dat_content = "MDO-BEZIER-OPTIMIZED\n" + "\n".join([f"  {x:.6f}  {y:.6f}" for x, y in opt_coords]) + "\n"
            st.download_button(
                label="Baixar Perfil Otimizado (.dat)",
                data=dat_content,
                file_name="perfil_MDO_otimizado.dat",
                mime="text/plain"
            )
            st.info("Importe este arquivo .dat diretamente no XFLR5 para validar a asa 3D completa.")
else:
    st.info("Faça o upload de um arquivo .dat na barra lateral esquerda (ex: S1223.dat) para começar.")
