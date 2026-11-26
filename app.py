import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os
from motor_mdo import load_selig_from_lines, run_optimization, run_xfoil_polar

st.set_page_config(page_title="Otimizador MDO Aerodesign", layout="wide")
st.title("Otimizador Paramétrico de Aerofólios (Bézier + XFOIL Multi-Re)")
st.markdown("Carregue um perfil baseline (`.dat`), defina a malha de Reynolds e as restrições estruturais/estabilidade do **MDO**.")

with st.sidebar:
    st.header("1. Perfil e Parametrização")
    uploaded_file = st.file_uploader("Upload do arquivo .dat (Selig)", type=["dat", "txt"])
    bezier_degree = st.select_slider(
        "Grau da Curva de Bézier (Pontos de Controle)",
        options=[6, 8, 10],
        value=8,
        help="Grau 6 = 10 variáveis | Grau 8 = 14 variáveis | Grau 10 = 18 variáveis"
    )

    st.header("2. Envelope de Reynolds (MDO)")
    re_min = st.number_input("Re Mínimo (Ponta / Decolagem)", value=120000, step=10000)
    re_max = st.number_input("Re Máximo (Raiz / Cruzeiro)", value=400000, step=10000)
    n_re = st.slider("Quantidade de Análises de Reynolds", 3, 9, 5, 1)
    re_list = [int(r) for r in np.linspace(re_min, re_max, n_re)]
    st.caption(f"Estações avaliadas: `{', '.join(map(str, re_list))}`")

    st.header("3. Restrições Geométricas e Estabilidade")
    min_max_tc = st.slider("Espessura Máx. Mínima (% corda)", 8.0, 18.0, 11.5, 0.5) / 100.0
    max_max_tc = st.slider("Espessura Máx. Limite (% corda)", 10.0, 22.0, 15.0, 0.5) / 100.0
    spar_x = st.slider("Posição da Longarina (% corda)", 15, 45, 30, 5) / 100.0
    min_spar_tc = st.slider("Espessura Mín. na Longarina (% corda)", 6.0, 16.0, 10.5, 0.5) / 100.0
    min_te_tc = st.number_input("Espessura Mín. Bordo de Fuga em 95%c", value=0.004, step=0.001, format="%.3f")
    min_cm = st.number_input("Limite de Momento Picador (Cm mín.)", value=-0.35, step=0.02)

    st.header("4. Objetivo da Missão")
    peso_cl = st.slider("Peso Sustentação Máxima (CL_max)", 0.0, 1.0, 0.65, 0.05)
    peso_ld = round(1.0 - peso_cl, 2)
    st.caption(f"Peso Eficiência Aerodinâmica (CL/CD): **{peso_ld:.2f}**")
    max_fev = st.slider("Máximo de Avaliações do Otimizador", 200, 1200, 450, 50)

def get_polar_from_coords(coords, re):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as tmp:
        tmp.write("EVAL-POLAR\n")
        for x, y in coords:
            tmp.write(f"  {x:.6f}  {y:.6f}\n")
        path = tmp.name
    pol = run_xfoil_polar(path, re, alpha_seq=(-2.0, 19.0, 1.0), ncrit=7)
    os.remove(path)
    return pol

if uploaded_file is not None:
    lines = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    upper_orig, lower_orig = load_selig_from_lines(lines)

    if st.button("Iniciar Otimização Multi-Reynolds", type="primary"):
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

        with st.spinner(f"Rodando simulações em {n_re} números de Reynolds com Bézier grau {bezier_degree}..."):
            base_coords, opt_coords, base_sc, opt_sc, history, (cp_u, cp_l) = run_optimization(
                upper_orig, lower_orig, re_list, mdo_specs, weights,
                bezier_degree=bezier_degree, max_fev=max_fev, progress_callback=update_ui
            )

        progress_bar.progress(1.0)
        st.success(f"Otimização concluída! Score subiu de **{base_sc:.4f}** para **{opt_sc:.4f}**.")

        col1, col2 = st.columns([2, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(9, 3.8))
            ax.plot(base_coords[:, 0], base_coords[:, 1], "k--", linewidth=1.5, label="Baseline (Bézier)")
            ax.plot(opt_coords[:, 0], opt_coords[:, 1], "b-", linewidth=2.2, label=f"Otimizado MDO (Grau {bezier_degree})")
            ax.scatter(cp_u[:, 0], cp_u[:, 1], color="red", s=22, zorder=5, label="Pontos de Controle")
            ax.scatter(cp_l[:, 0], cp_l[:, 1], color="red", s=22, zorder=5)
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
            st.info(f"Otimizado simultaneamente em {n_re} estações de Reynolds ({re_min:,} a {re_max:,}).")

        re_central = re_list[len(re_list) // 2]
        st.subheader(f"Comparativo Aerodinâmico 2D na Estação Central (Re = {re_central:,})")
        pol_base = get_polar_from_coords(base_coords, re_central)
        pol_opt = get_polar_from_coords(opt_coords, re_central)

        if pol_base is not None and pol_opt is not None:
            p1, p2 = st.columns(2)
            with p1:
                fig_cl, ax_cl = plt.subplots(figsize=(6, 4))
                ax_cl.plot(pol_base[:, 0], pol_base[:, 1], "k--", label=f"Baseline (CLmax={np.max(pol_base[:,1]):.2f})")
                ax_cl.plot(pol_opt[:, 0], pol_opt[:, 1], "b-", linewidth=2, label=f"Otimizado (CLmax={np.max(pol_opt[:,1]):.2f})")
                ax_cl.set_title("Sustentação (CL x Alpha)")
                ax_cl.set_xlabel("Alpha (graus)")
                ax_cl.set_ylabel("CL")
                ax_cl.grid(True, linestyle=":", alpha=0.6)
                ax_cl.legend()
                st.pyplot(fig_cl)

            with p2:
                fig_ld, ax_ld = plt.subplots(figsize=(6, 4))
                ax_ld.plot(pol_base[:, 2], pol_base[:, 1], "k--", label="Baseline")
                ax_ld.plot(pol_opt[:, 2], pol_opt[:, 1], "b-", linewidth=2, label="Otimizado")
                ax_ld.set_title("Polar de Arrasto (CL x CD)")
                ax_ld.set_xlabel("CD")
                ax_ld.set_ylabel("CL")
                ax_ld.grid(True, linestyle=":", alpha=0.6)
                ax_ld.legend()
                st.pyplot(fig_ld)
else:
    st.info("Faça o upload de um arquivo .dat na barra lateral esquerda (ex: S1223.dat) para começar.")