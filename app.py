import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


section = st.sidebar.radio("Navigați la:",
                           ["Date proiect", "Introducere date si afisarea lor",
                            "Selectare coloane din tabel","Preprocesare date","Prelucrari statistice"])

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
        #citim fisierul
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1",)
        #buton afisare tabel
    if st.button(" Afisare tabel "):
        st.dataframe(df)#afiseaza tabelul la apasarea butonului
        st.success(":D DOAMNE ia uite ce tabel frumos <3!")


    if st.button(" Afiseaza graficul lansarilor "):
        if "Company Name" in df.columns and "Launched Year" in df.columns:
            #Grupam datele pentru a numara modelele per producator si an
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

        #citim fisierul
        df = pd.read_csv(uploaded_file, encoding="ISO-8859-1",)

        all_columns = df.columns.tolist()
        #afisare coloane
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
                st.success(f"Coloana '{selected_cat_col}' a fost codificata si adăugata ca '{selected_cat_col}_Encoded'")
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
        agg_func = st.multiselect("Selecteaza functiile de agregare:", ["mean", "sum", "min", "max", "count", "median", "std"])

        if st.button("Aplic gruparea si agregarea"):
            if group_col and agg_col and agg_func:
                grouped_df = df.groupby(group_col)[agg_col].agg(agg_func)
                st.write(" Rezultatul gruparii si agregarii:")
                st.dataframe(grouped_df)
            else:
                st.warning("Selecteaza toate opțiunile necesare pentru a aplica operatiile.")

        st.subheader("3️. Functii de grup (apply, transform, filter)")

        func_opt = st.selectbox("Alege o functie de grup:", ["Media per grup (transform)", "Filtrare grupuri (filter)", "Prelucrare personalizată (apply)"])

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

            df[selected_num_col + "_diff_grup"] = df.groupby(group_col).apply(lambda x: custom_function(x)).reset_index(level=0, drop=True)
            st.success("Coloana cu diferente fata de media grupului adăugata.")
            st.dataframe(df[[group_col, selected_num_col, selected_num_col + "_diff_grup"]].head())



