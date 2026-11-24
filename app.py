    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Missing values", int(data.isna().sum().sum()))

    selected_crop = st.selectbox("Filter by crop", ["All crops"] + sorted(data["label"].unique()))
    filtered = data if selected_crop == "All crops" else data[data["label"] == selected_crop]

    st.markdown("### Feature summary")
    st.dataframe(filtered[FEATURES].describe().T.round(2), use_container_width=True)

    st.markdown("### Compare feature distributions")
    feature = st.selectbox("Choose a feature", FEATURES, format_func=lambda x: DISPLAY_NAMES[x])
    fig, ax = plt.subplots(figsize=(10, 4))
    if selected_crop == "All crops":
        filtered.boxplot(column=feature, by="label", ax=ax, rot=70, grid=False)
        fig.suptitle("")
        ax.set_xlabel("Crop")
    else:
        ax.hist(filtered[feature], bins=18, color="#3a9562", edgecolor="white")
        ax.set_xlabel(DISPLAY_NAMES[feature])
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Dataset preview")
    st.dataframe(filtered.head(100), use_container_width=True)
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered data as CSV", csv, "agromitra_dataset.csv", "text/csv")


# --------------------------- MODEL INSIGHTS ---------------------------
elif page == "🧠 Model Insights":
    st.subheader("🧠 Model performance and explainability")
    st.write("A Random Forest classifier is trained when the app starts, using an 80/20 stratified train-test split.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{accuracy*100:.2f}%")
    c2.metric("Training rows", f"{train_count:,}")
    c3.metric("Testing rows", f"{test_count:,}")

    st.markdown("### Feature importance")
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    importance.plot(kind="barh", ax=ax, color="#3a9562")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("### Per-crop evaluation")
    report_df = pd.DataFrame(report).T
    report_df = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    st.dataframe(report_df.round(3), use_container_width=True)

    with st.expander("Show confusion matrix"):
        st.caption("Rows are actual crops; columns are predicted crops.")
        fig, ax = plt.subplots(figsize=(11, 9))
        image = ax.imshow(cm, interpolation="nearest", cmap="Greens")
        ax.set_title("Confusion matrix")
        ax.set_xlabel("Predicted crop")
        ax.set_ylabel("Actual crop")
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels([crop_title(x) for x in classes], rotation=90, fontsize=7)
        ax.set_yticklabels([crop_title(x) for x in classes], fontsize=7)
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# --------------------------- ABOUT ---------------------------
elif page == "ℹ️ About":
    st.subheader("About Agro Mitra")
    st.write("""
    Agro Mitra is a prototype crop recommendation application built around the supplied crop recommendation dataset.
    It uses soil nutrient values and climate measurements to estimate which crop label best matches the input pattern.
    """)
    st.markdown("### Dataset columns")
    st.dataframe(pd.DataFrame({
        "Column": FEATURES + ["label"],
        "Meaning": [
            "Nitrogen value", "Phosphorus value", "Potassium value",
            "Temperature", "Relative humidity", "Soil pH", "Rainfall", "Crop class / target"
        ],
        "Role": ["Input"] * 7 + ["Target"],
    }), use_container_width=True)

    st.markdown("### Important limitations")
    st.markdown("""
    - The dataset does not specify the units for N, P and K. Use values on a consistent scale.
    - The dataset contains crop labels, not yield, profit, disease or fertilizer recommendations.
    - This version does not fetch live weather, mandi prices or government scheme information.
    - Predictions reflect the dataset and model; they should be validated with local agricultural experts.
    """)
    st.markdown("### Technology stack")
    st.write("Python · Streamlit · Pandas · NumPy · Scikit-learn · Matplotlib")
    st.markdown("### Dataset integrity")
    st.write(f"Loaded **{len(data):,}** valid records across **{data['label'].nunique()}** crop classes.")
