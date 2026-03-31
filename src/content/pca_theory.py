import streamlit as st


def render_pca_theory_panel():
    with st.expander("📐 What is PCA and how does it work?", expanded=False):
        st.markdown(
            """
            <div style="margin-bottom: 16px; color: #94a3b8; font-size: 0.98rem;">
                The intuition, the math, and when to use it.
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                """
                <div style="
                    border: 1px solid rgba(56,189,248,0.35);
                    border-radius: 18px;
                    padding: 18px;
                    background: rgba(56,189,248,0.06);
                    min-height: 220px;
                ">
                    <div style="font-size: 1.05rem; font-weight: 800; color: #38bdf8; margin-bottom: 12px;">
                        Step 1 · Standardise
                    </div>
                    <div style="color: #94a3b8; line-height: 1.7; font-size: 0.98rem;">
                        Centre each feature to mean = 0 and scale to std = 1, so no feature dominates just because of its units.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"z = \frac{x - \mu}{\sigma}")

        with col2:
            st.markdown(
                """
                <div style="
                    border: 1px solid rgba(168,85,247,0.35);
                    border-radius: 18px;
                    padding: 18px;
                    background: rgba(168,85,247,0.06);
                    min-height: 220px;
                ">
                    <div style="font-size: 1.05rem; font-weight: 800; color: #a855f7; margin-bottom: 12px;">
                        Step 2 · Covariance Matrix
                    </div>
                    <div style="color: #94a3b8; line-height: 1.7; font-size: 0.98rem;">
                        Build a matrix that captures how each pair of features vary together. Large values indicate strong shared movement.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"C = \frac{1}{n-1}X^T X")

        with col3:
            st.markdown(
                """
                <div style="
                    border: 1px solid rgba(74,222,128,0.35);
                    border-radius: 18px;
                    padding: 18px;
                    background: rgba(74,222,128,0.06);
                    min-height: 220px;
                ">
                    <div style="font-size: 1.05rem; font-weight: 800; color: #4ade80; margin-bottom: 12px;">
                        Step 3 · Eigenvectors
                    </div>
                    <div style="color: #94a3b8; line-height: 1.7; font-size: 0.98rem;">
                        Decompose the covariance matrix into directions (eigenvectors) and magnitudes (eigenvalues). The largest eigenvalue captures the most variance.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"Cv = \lambda v")

        st.markdown(
            """
            <div style="
                margin-top: 14px;
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 18px;
                padding: 18px;
                background: rgba(255,255,255,0.03);
                color: #94a3b8;
                line-height: 1.7;
                font-size: 1rem;
            ">
                <strong style="color: #e5e7eb;">The result:</strong>
                You get a new coordinate system where <strong style="color: #60a5fa;">PC1</strong> points in the direction of highest variance,
                <strong style="color: #60a5fa;">PC2</strong> in the next-highest (orthogonal to PC1), and so on.
                You can often drop the later PCs without losing much information — that is dimensionality reduction.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="
                margin-top: 18px;
                border: 1px solid rgba(245,158,11,0.28);
                border-radius: 20px;
                padding: 20px;
                background: rgba(245,158,11,0.05);
            ">
                <div style="font-size: 1.12rem; font-weight: 800; color: #fbbf24; margin-bottom: 14px;">
                    🔢 Eigenvalues & Eigenvectors — the math behind PCA
                </div>
                <div style="color: #94a3b8; line-height: 1.7; font-size: 0.98rem; margin-bottom: 14px;">
                    An <strong style="color: #e5e7eb;">eigenvector</strong> of the covariance matrix
                    <strong style="color: #fbbf24;"> C </strong>
                    is a special direction
                    <strong style="color: #fbbf24;"> v </strong>
                    that does not rotate when
                    <strong style="color: #fbbf24;"> C </strong>
                    is applied — it only stretches. The stretch factor is the
                    <strong style="color: #e5e7eb;">eigenvalue</strong>
                    <strong style="color: #fbbf24;"> λ </strong>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.latex(r"C \cdot v = \lambda \cdot v")

        st.caption(
            "C = covariance matrix · v = eigenvector (principal component direction) · λ = eigenvalue (variance along that direction)"
        )

        s1, s2, s3 = st.columns(3)

        with s1:
            st.markdown(
                """
                <div style="
                    border: 1px solid rgba(245,158,11,0.22);
                    border-radius: 18px;
                    padding: 18px;
                    background: rgba(0,0,0,0.14);
                    min-height: 210px;
                ">
                    <div style="font-size: 1.05rem; font-weight: 800; color: #fbbf24; margin-bottom: 10px;">
                        Step 1 — Form C
                    </div>
                    <div style="color: #94a3b8; line-height: 1.7; font-size: 0.97rem;">
                        The covariance matrix encodes how every pair of features varies together. It is symmetric and square.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"C = \sum (x_i - \bar{x})(x_i - \bar{x})^T")

        with s2:
            st.markdown(
                """
                <div style="
                    border: 1px solid rgba(245,158,11,0.22);
                    border-radius: 18px;
                    padding: 18px;
                    background: rgba(0,0,0,0.14);
                    min-height: 210px;
                ">
                    <div style="font-size: 1.05rem; font-weight: 800; color: #fbbf24; margin-bottom: 10px;">
                        Step 2 — Decompose C
                    </div>
                    <div style="color: #94a3b8; line-height: 1.7; font-size: 0.97rem;">
                        Solve for all eigenvector/eigenvalue pairs. Each eigenvector becomes a principal component direction.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"C e_i = \lambda_i e_i")

        with s3:
            st.markdown(
                """
                <div style="
                    border: 1px solid rgba(245,158,11,0.22);
                    border-radius: 18px;
                    padding: 18px;
                    background: rgba(0,0,0,0.14);
                    min-height: 210px;
                ">
                    <div style="font-size: 1.05rem; font-weight: 800; color: #fbbf24; margin-bottom: 10px;">
                        Step 3 — Select top k
                    </div>
                    <div style="color: #94a3b8; line-height: 1.7; font-size: 0.97rem;">
                        Sort eigenvalues in descending order. Keep the top k eigenvectors — they capture the most variance.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.latex(r"\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d")

        st.markdown(
            """
            <div style="
                margin-top: 16px;
                border-top: 1px solid rgba(245,158,11,0.18);
                padding-top: 16px;
                color: #94a3b8;
                line-height: 1.7;
                font-size: 0.98rem;
            ">
                <strong style="color: #fbbf24;">Why eigenvalues = variance:</strong>
                PCA looks for directions <em>v</em> that maximise projected spread, i.e. maximise
                <strong style="color: #e5e7eb;"> vᵀCv </strong>.
                Substituting
                <strong style="color: #e5e7eb;"> Cv = λv </strong>
                gives
                <strong style="color: #e5e7eb;"> vᵀ(λv) = λ‖v‖² = λ </strong>
                for unit vectors, so the eigenvalue is exactly the variance along that principal direction.
            </div>
            """,
            unsafe_allow_html=True,
        )