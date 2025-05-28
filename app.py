# Pandas și NumPy
import pandas as pd
import numpy as np

# Preprocesare date
from sklearn.preprocessing import StandardScaler

# Algoritm de clusterizare
from sklearn.cluster import KMeans

# Reducere dimensionalitate pentru vizualizare
from sklearn.decomposition import PCA

# Vizualizare
import matplotlib.pyplot as plt

# Streamlit (pentru interfața aplicației)
import streamlit as st

import statsmodels.api as sm

section = st.sidebar.radio("Navigați la:",
                           ["Date proiect", "Introducere date si afisarea lor",
                            "Selectare coloane din tabel", "Preprocesare date", "Prelucrari statistice",
                            "Clusterizare (KMeans)", "Regresie logistica", "Regresie multipla"])

if section == "Date proiect":
    st.header("Proiect Pachete Software")
    st.markdown(
        """
           <style>
           .custom-title {
               color: #A67C52 !Important;
               font-size: 40px;
               text-align: center;
           }

           .custom-header {
               color: #A67C52 !Important;
               font-size: 40px;
               text-align: center;
           }
           .custom-header2 {
               color: #5A3E6F !Important;
               font-size: 40px;
               text-align: left;
           }
           .custom-header4 {
               color: #A67C52 !Important;
               font-size: 40px;
               text-align: left;
           }
           </style>
           """,
        unsafe_allow_html=True)

    st.markdown('<h1 class="custom-title">Set de date despre telefoane mobile</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="custom-header2">Descrierea datelor</h1>', unsafe_allow_html=True)
    st.markdown("""

       Acest set de date conține specificații detaliate și prețuri oficiale de lansare ale diferitelor modele de telefoane mobile de la diverse companii. Oferă informații despre hardware-ul smartphone-urilor, tendințele de preț și competitivitatea mărcilor în mai multe țări. Setul de date include caracteristici cheie, cum ar fi memoria RAM, specificațiile camerei, capacitatea bateriei, detaliile procesorului și dimensiunea ecranului.

    Un aspect important al acestui set de date este informația despre prețuri. Prețurile înregistrate reprezintă prețurile oficiale de lansare ale telefoanelor mobile la momentul introducerii lor pe piață. Prețurile variază în funcție de țară și perioada de lansare, ceea ce înseamnă că modelele mai vechi reflectă prețurile lor originale de lansare, în timp ce modelele mai noi includ cele mai recente prețuri de lansare. Acest lucru face ca setul de date să fie valoros pentru studierea tendințelor de preț în timp și pentru compararea accesibilității smartphone-urilor în diferite regiuni.

    """
                , unsafe_allow_html=True)

    st.markdown('<h3 class="custom-header2">Caracteristici: </h3>', unsafe_allow_html=True)
    st.markdown(r"""

    -**Numele Companiei:** Marca sau producătorul telefonului mobil.

    -**Numele Modelului:** Modelul specific al smartphone-ului.

    -**Greutatea Mobilului:** Greutatea telefonului mobil (în grame).

    -**RAM:** Cantitatea de Memorie cu Acces Aleatoriu (RAM) din dispozitiv (în GB).

    -**Camera Frontală:** Rezoluția camerei frontale (pentru selfie) (în MP).

    -**Camera Spate:** Rezoluția camerei principale din spate (în MP).

    -**Procesor:** Chipset-ul sau procesorul utilizat în dispozitiv.

    -**Capacitatea Bateriei:** Dimensiunea bateriei smartphone-ului (în mAh).

    -**Dimensiunea Ecranului:** Dimensiunea afișajului smartphone-ului (în inci).

    -**Prețul de Lansare:** (Pakistan, India, China, SUA, Dubai): Prețul oficial de lansare al mobilului în țara respectivă la momentul lansării sale. Prețurile variază în funcție de anul lansării mobilului.

    -**Anul Lansării:** Anul în care telefonul mobil a fost lansat oficial.

    """, unsafe_allow_html=True)
elif section == "Introducere date si afisarea lor":

    st.markdown(
        """
        <style>
        .custom-title {
        color: #A67C52 !Important;
        font-size: 40px;
        text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True)
    # File uploader
    st.markdown('<h4 class="custom-title">Introduceti datele: </h4>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Incarca un fisier", type=["csv"])
    if uploaded_file is not None:
        st.write("Fișier incarcat:", uploaded_file.name)
        # citim fisierul
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", )
        # buton afisare tabel
    if st.button(" Afisare tabel "):
        st.dataframe(df)  # afiseaza tabelul la apasarea butonului
        st.success(":D DOAMNE ia uite ce tabel frumos <3!")

    if st.button(" Afiseaza graficul lansarilor "):
        if "Company Name" in df.columns and "Launched Year" in df.columns:
            # Grupam datele pentru a numara modelele per producator si an
            phone_counts = df.groupby(["Launched Year", "Company Name"]).size().reset_index(name="Numar modele")

            # Cream un pivot table pentru vizualizare mai clara
            pivot_table = phone_counts.pivot(index="Launched Year", columns="Company Name",
                                             values="Numar modele").fillna(0)
            print(pivot_table['Launched Year'])
            # Afișam graficul
            st.write(" **Evolutia lansarilor de telefoane pe producator**")
            st.line_chart(pivot_table)

            st.success(":D Graficul a fost generat cu succes!")
        else:
            st.error(":_( Coloanele nu au fost gasite...")

    # Text area Observatii
    user_text_area = st.text_area("Observatii cu privire la outputul primit:")
    st.write("Observatii:", user_text_area)

elif section == "Selectare coloane din tabel":
    st.markdown(
        """
           <style>
           .custom-title {
               color: #A67C52 !Important;
               font-size: 40px;
               text-align: left;
           }
           </style>
           """,
        unsafe_allow_html=True)

    st.markdown('<h5 class = "custom-title">Incarcati fisierul</h5>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Incarcă un fisier", type=["csv"])

    if uploaded_file is not None:
        st.write("Fisier încarcat:", uploaded_file.name)

        # citim fisierul
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1", )

        all_columns = df.columns.tolist()
        # afisare coloane
        st.write("**Coloane disponibile:**", all_columns)

        # Selector pentru coloanele pe care utilizatorul vrea să le vada
        selected_columns = st.multiselect("Selecteaza coloanele de afisat:", all_columns)

    # Buton pentru afișarea tabelului
    if st.button("Afiseaza tabelul"):
        if selected_columns:
            st.write("**Tabelul cu datele selectate:**")
            st.table(df[selected_columns])  # Afișeaza doar coloanele selectate


elif section == "Preprocesare date":
    st.markdown(
        """
           <style>
           .custom-title {
               color: #A67C52 !Important;
               font-size: 40px;
               text-align: center;
           }

           .custom-header {
               color: #A67C52 !Important;
               font-size: 40px;
               text-align: center;
           }
           .custom-header2 {
               color: #5A3E6F !Important;
               font-size: 40px;
               text-align: left;
           }
           .custom-header4 {
               color: #A67C52 !Important;
               font-size: 40px;
               text-align: left;
           }
           </style>
           """,
        unsafe_allow_html=True)
    st.markdown('<h6 class="custom-title">Preprocesare date: valori lipsă și extreme</h6>', unsafe_allow_html=True)
    st.markdown('<h7 class = "custom-header2">Incarcati fisierul</h7>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Incarca un fisier CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")

        st.subheader("1️.Valori lipsa in setul de date")
        st.write(df.isnull().sum())

        actiune = st.radio("Alege o actiune pentru valorile lipsa:",
                           ["Nicio actiune", "Completare cu media", "Stergere randuri"])
        if actiune == "Completare cu media":
            df.fillna(df.mean(numeric_only=True), inplace=True)
            st.success("Valorile lipsa au fost completate cu media coloanelor numerice.")
        elif actiune == "Stergere randuri":
            df.dropna(inplace=True)
            st.success("Randurile cu valori lipsa au fost eliminate.")
        #
        st.subheader("2️.Tratarea valorilor extreme (outlieri)")
        col_numeric = st.selectbox("Selecteaza o coloana numerica:",
                                   df.select_dtypes(include=np.number).columns.tolist())
        if col_numeric:
            Q1 = df[col_numeric].quantile(0.25)
            Q3 = df[col_numeric].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col_numeric] < Q1 - 1.5 * IQR) | (df[col_numeric] > Q3 + 1.5 * IQR)]

            st.write(f" Valori extreme in coloana **{col_numeric}**:")
            st.dataframe(outliers)

            df_filtered = df[(df[col_numeric] >= Q1 - 1.5 * IQR) & (df[col_numeric] <= Q3 + 1.5 * IQR)]
            st.write(f" Date fara valori extreme in coloana **{col_numeric}**:")
            st.dataframe(df_filtered)

        #
        st.subheader("3️.Codificarea variabilelor categoriale")

        cat_columns = df.select_dtypes(include=['object']).columns.tolist()
        selected_cat_col = st.selectbox("Selecteaza o coloana categoriala pentru codificare:", cat_columns)

        encoding_type = st.radio("Alege tipul de codificare:", ["Label Encoding", "One-Hot Encoding"])

        if st.button("Aplica codificarea"):
            if encoding_type == "Label Encoding":
                from sklearn.preprocessing import LabelEncoder

                le = LabelEncoder()
                df[selected_cat_col + "_Encoded"] = le.fit_transform(df[selected_cat_col].astype(str))
                st.success(
                    f"Coloana '{selected_cat_col}' a fost codificata si adăugata ca '{selected_cat_col}_Encoded'")
                st.write(df[[selected_cat_col, selected_cat_col + "_Encoded"]].head())
            elif encoding_type == "One-Hot Encoding":
                df_encoded = pd.get_dummies(df, columns=[selected_cat_col])
                st.success(f"One-Hot Encoding aplicat pe '{selected_cat_col}'.")
                st.write(df_encoded.head())
        #
        st.subheader("4️. Scalarea datelor numerice")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected_scaling_cols = st.multiselect("Selecteaza coloanele numerice pentru scalare:", numeric_cols)

        scaling_method = st.radio("Alege metoda de scalare:", ["Min-Max Scaling", "Standard Scaling"])

        if st.button("Aplică scalarea"):
            from sklearn.preprocessing import MinMaxScaler, StandardScaler

            if selected_scaling_cols:
                if scaling_method == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()

                df_scaled = df.copy()
                df_scaled[selected_scaling_cols] = scaler.fit_transform(df[selected_scaling_cols])

                st.success(f"Scalarea '{scaling_method}' a fost aplicata cu succes!")
                st.write(df_scaled[selected_scaling_cols].head())
            else:
                st.warning("Selecteaza cel putin o coloana pentru a aplica scalarea.")

elif section == "Prelucrari statistice":
    st.markdown('<h8 style="color:#A67C52;"> Prelucrari statistice, grupari si agregari</h8>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Incarcă un fisier CSV", type=["csv"], key="statistica")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        st.write(":D Date incarcate:")
        st.dataframe(df.head())

        st.subheader("1️. Statistici descriptive generale")
        st.write(df.describe(include='all'))

        st.subheader("2️. Grupare si agregare")
        cat_columns = df.select_dtypes(include='object').columns.tolist()
        num_columns = df.select_dtypes(include=np.number).columns.tolist()

        group_col = st.selectbox("Selecteaza coloana pentru grupare:", cat_columns)
        agg_col = st.multiselect("Selecteaza coloanele numerice pentru agregare:", num_columns)
        agg_func = st.multiselect("Selecteaza functiile de agregare:",
                                  ["mean", "sum", "min", "max", "count", "median", "std"])

        if st.button("Aplic gruparea si agregarea"):
            if group_col and agg_col and agg_func:
                grouped_df = df.groupby(group_col)[agg_col].agg(agg_func)
                st.write(" Rezultatul gruparii si agregarii:")
                st.dataframe(grouped_df)
            else:
                st.warning("Selecteaza toate opțiunile necesare pentru a aplica operatiile.")

        st.subheader("3️. Functii de grup (apply, transform, filter)")

        func_opt = st.selectbox("Alege o functie de grup:", ["Media per grup (transform)", "Filtrare grupuri (filter)",
                                                             "Prelucrare personalizată (apply)"])

        if func_opt == "Media per grup (transform)":
            if group_col and agg_col:
                transformed_df = df.copy()
                for col in agg_col:
                    transformed_df[col + "_grup_mean"] = df.groupby(group_col)[col].transform('mean')
                st.success("Media per grup a fost adăugata.")
                st.write(transformed_df[[group_col] + agg_col + [col + "_grup_mean" for col in agg_col]].head())

        elif func_opt == "Filtrare grupuri (filter)":
            min_count = st.number_input("Filtreaza grupurile care au cel putin N elemente:", min_value=1, value=5)
            filtered_df = df.groupby(group_col).filter(lambda x: len(x) >= min_count)
            st.success(f"Grupurile cu cel putin {min_count} observatii au fost pastrate.")
            st.dataframe(filtered_df)

        elif func_opt == "Prelucrare personalizata (apply)":
            st.info("Exemplu: diferenta dintre valoare si media grupului")
            selected_num_col = st.selectbox("Coloana numerica pentru prelucrare:", num_columns)


            def custom_function(x):
                return x[selected_num_col] - x[selected_num_col].mean()


            df[selected_num_col + "_diff_grup"] = df.groupby(group_col).apply(lambda x: custom_function(x)).reset_index(
                level=0, drop=True)
            st.success("Coloana cu diferente fata de media grupului adăugata.")
            st.dataframe(df[[group_col, selected_num_col, selected_num_col + "_diff_grup"]].head())

elif section == "Clusterizare (KMeans)":
    st.markdown('<h4 style="color:#5A3E6F;">Clusterizare KMeans</h4>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Incarca fisierul CSV", type=["csv"], key="kmeans")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        st.write("Datele au fost incarcate.")
        st.dataframe(df.head())

        # Redenumire coloane
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

        # Curatare si conversie Launched_Price_USA daca exista
        if "Launched_Price_USA" in df.columns:
            df["Launched_Price_USA"] = (
                df["Launched_Price_USA"]
                .astype(str)
                .str.replace("USD", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
                .astype(float)
            )

        # Selectam doar coloanele numerice utile (excludem Launched_Year daca e singura)
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        filtered_cols = [col for col in num_cols if df[col].nunique() > 2]  # eliminam coloane cu variabilitate mica
        selected_cols = st.multiselect("Selecteaza coloanele numerice pentru clusterizare:", filtered_cols)

        if selected_cols:
            if len(selected_cols) < 2:
                st.warning("Pentru vizualizare PCA este necesar sa selectezi cel putin 2 coloane numerice.")
            else:
                k = st.slider("Alege numarul de clustere:", min_value=2, max_value=10, value=3)

                from sklearn.preprocessing import StandardScaler
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
                import matplotlib.pyplot as plt

                X = df[selected_cols].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
                clusters = kmeans.fit_predict(X_scaled)

                df_result = X.copy()
                df_result["Cluster"] = clusters

                st.subheader("Rezultatele clusterizarii:")
                st.dataframe(df_result)

                # PCA si afisare grafic
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_scaled)
                df_result["PCA1"] = components[:, 0]
                df_result["PCA2"] = components[:, 1]

                fig, ax = plt.subplots()
                colors = plt.cm.get_cmap("tab10", k)

                for cluster_id in sorted(df_result["Cluster"].unique()):
                    cluster_data = df_result[df_result["Cluster"] == cluster_id]
                    ax.scatter(cluster_data["PCA1"], cluster_data["PCA2"],
                               label=f"Cluster {cluster_id}",
                               color=colors(cluster_id))

                ax.set_title("Vizualizare 2D a clusterelor")
                ax.set_xlabel("PCA1")
                ax.set_ylabel("PCA2")
                ax.legend()
                st.pyplot(fig)
        else:
            st.warning("Selecteaza cel putin doua coloane numerice cu variabilitate suficienta.")


elif section == "Regresie logistica":
    st.markdown('<h4 style="color:#5A3E6F;">Regresie logistica (clasificare telefoane)</h4>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Incarca fisierul CSV", type=["csv"], key="logistic")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        st.write("Datele au fost incarcate.")
        st.dataframe(df.head())

        # Redenumim coloanele pentru a evita probleme
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

        if "Launched_Price_USA" in df.columns:
            # Curatam si convertim coloana la float
            df["Launched_Price_USA"] = (
                df["Launched_Price_USA"]
                .astype(str)
                .str.replace("USD", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
                .astype(float)
            )

            # Eticheta binara: 1 daca pretul este peste mediana, 0 altfel
            price_median = df["Launched_Price_USA"].median()
            df["High_Price"] = (df["Launched_Price_USA"] > price_median).astype(int)
            st.write(f"Eticheta binara creata: 1 daca Launched_Price_USA > {price_median}, altfel 0")

            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            num_cols.remove("Launched_Price_USA")
            if "High_Price" in num_cols:
                num_cols.remove("High_Price")

            selected_features = st.multiselect("Selecteaza coloanele pentru predictie:", num_cols)

            if selected_features:
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.metrics import accuracy_score, classification_report

                X = df[selected_features].fillna(0)
                y = df["High_Price"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = LogisticRegression()
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                acc = accuracy_score(y_test, y_pred)

                st.write(f"Acuratetea modelului pe datele de test: {acc:.2f}")
                st.text("Raport de clasificare:")
                st.text(classification_report(y_test, y_pred))
            else:
                st.warning("Selecteaza cel putin o coloana numerica pentru a construi modelul.")
        else:
            st.error("Coloana 'Launched_Price_USA' nu a fost gasita in setul de date.")

elif section == "Regresie multipla":
    st.markdown('<h4 style="color:#5A3E6F;">Regresie liniara multipla (statsmodels)</h4>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Incarca fisierul CSV", type=["csv"], key="regresie_multipla")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        st.write("Datele au fost incarcate.")
        st.dataframe(df.head())

        # Redenumim coloanele
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

        # Daca exista Launched_Price_USA, incercam sa-l convertim
        if "Launched_Price_USA" in df.columns:
            df["Launched_Price_USA"] = (
                df["Launched_Price_USA"]
                .astype(str)
                .str.replace("USD", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
                .astype(float)
            )

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        target_col = st.selectbox("Selecteaza coloana dependenta (y):", numeric_cols)
        feature_cols = st.multiselect("Selecteaza variabilele explicative (X):",
                                      [col for col in numeric_cols if col != target_col])

        if st.button("Aplica regresia"):
            if feature_cols and target_col:

                X = df[feature_cols].fillna(0)
                y = df[target_col]

                X = sm.add_constant(X)  # Adaugam constanta pentru intercept
                model = sm.OLS(y, X).fit()

                st.subheader("Rezultatele modelului:")
                st.text(model.summary())
                st.success("Modelul a fost estimat cu succes.")
            else:
                st.warning("Selecteaza cel putin o variabila explicativa si o variabila dependenta.")
