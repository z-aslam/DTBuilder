#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import base64
import time
from streamlit_tags import st_tags
sns.set(
    { "figure.figsize": (6, 4) },
    style='ticks',
    color_codes=True,
    font_scale=0.8
)
#%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys
#if 'google.colab' in sys.modules:
#    !pip install -q dtreeviz
import dtreeviz
st.set_page_config(layout="wide")
from streamlit_float import *
import configparser
from ipywidgets import interact
import ipywidgets as widgets
from IPython.display import display
import streamlit as st
from sklearn.model_selection import cross_val_score
from sklearn.metrics import *
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from io import StringIO
import ast
from datetime import datetime
import streamlit.components.v1 as components
import base64
import json
# initialize float feature/capability
float_init()
uploaded_file=None
tmp_file_name = './saves/'+datetime.today().strftime('%Y-%m-%d %H:%M')+'.ini'
config = configparser.ConfigParser()
if 'config_write' not in st.session_state:
    st.session_state['config_write'] = ''

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

# if 'file_uploaded' not in st.session_state:
#     st.session_state.file_uploaded = False

css = '''
<style>
    [data-testid='stFileUploader'] {
        width: max-content;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        padding-top: 0;
        margin-top:0;
    }

</style>
'''

st.markdown(css, unsafe_allow_html=True)
def save(conf):
    st.session_state['config_write'] = conf

with st.container():
    title_cols = st.columns((10,2,3))
    with title_cols[0]:
        st.title("Classification DT Builder")
    with title_cols[1]:
        if st.button("Save", key="save_button", disabled= not st.session_state.button_clicked):
            with open(tmp_file_name,'w') as configfile:
                st.session_state['config_write'].write(configfile)
                st.write("Added to /saves directory as:")
                st.write(tmp_file_name)
        #    st.write(dict(st.session_state['config_write']["VARIABLES"]))

    with title_cols[2]:
        # if st.button("Upload"):
        uploaded_config = st.file_uploader("Upload File", type=["ini"],label_visibility="collapsed")

col = st.columns((1.5, 4.5, 2), gap='large')
 
# "df": 0,
# "X": 0, 
# "y": 0,

config_dict = {
    "target_dropdown": 0,
    "max_depth_slider": 5,
    "criterion_dropdown": 0,
    "splitter_dropdown": 0,
    "min_samples_split_slider": 2,
    "min_samples_leaf_slider": 1,
    "test_split_slider": 0.25,
    "cross_val": 5,
    "scoring_index": 1,
    "max_depth_inputs2": [],
    "min_samples_inputs2": []
}

def svg_write(svg, center=True):
    """
    Disable center to left-margin align like other objects.
    """
    # Encode as base 64
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = f'<p style="text-align:center; display: flex; justify-content: {css_justify};">'
    html = f'{css}<img src="data:image/svg+xml;base64,{b64}"/>'

    # Write the HTML
    st.write(html, unsafe_allow_html=True)

def update_dropdowns(df):
    # Get list of columns from the dataframe
    columns = df.columns.tolist()
    # Update the dropdown options
    config_dict["target_dropdown"] = st.selectbox(
    "Choose target column: ",
    columns, index = config_dict["target_dropdown"])
    st.write("You selected:", config_dict["target_dropdown"])

    # Optionally, pre-select the last column as the target (common convention)
    #config_dict["target_dropdown"].value = columns[-1]
    #target = config_dict["target_dropdown"].value

    X, y = df.drop(columns=config_dict["target_dropdown"]), df[config_dict["target_dropdown"]]
    X.head(5)
    display(config_dict["target_dropdown"])
    #print("Select the target column.")
    target = config_dict["target_dropdown"]

    st.subheader("Configuration", divider = 'grey')

def load_data(file):
    data = pd.read_csv(file)
    return data
df = None

a = time.time()
with col[0]:
    st.subheader("Configurator", divider='grey')
    uploaded_file = st.file_uploader("Choose a CSV file ", type=["csv"])
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    if uploaded_file:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("File size is too big, max file size is 200mb!")
    # uploaded_config = 
    # creating final config dictionary
    # opening ini file (w means create new if not aval)
    if uploaded_config:
        parser = configparser.ConfigParser()
        stringio = StringIO(uploaded_config.getvalue().decode("utf-8"))
        parser.read_string(stringio.read())
        # st.write(dict(parser["VARIABLES"]))
        st.write("Config Loaded")
        config_dict = dict(parser["VARIABLES"])
        config_dict["max_depth_slider"] = int(config_dict["max_depth_slider"])
        config_dict["min_samples_split_slider"] = int(config_dict["min_samples_split_slider"])
        config_dict["min_samples_leaf_slider"] = int(config_dict["min_samples_leaf_slider"])
        config_dict["test_split_slider"] = float(config_dict["test_split_slider"])
        config_dict["cross_val"] = int(config_dict["cross_val"])
        config_dict["scoring_index"] = int(config_dict["scoring_index"])

        config_dict["max_depth_inputs2"] = ast.literal_eval(config_dict["max_depth_inputs2"])
        config_dict["min_samples_inputs2"] = ast.literal_eval(config_dict["min_samples_inputs2"])

        uploaded_file = StringIO(dict(parser["DATAFRAME"])["df"])

        
        # for section in config.sections():
        #     items = config.items(section)
        #     ini_dict[section] = dict(items)
        # st.write(config.sections())
        # config_dict = ini_dict["DEFAULT"]

if uploaded_file is None:
    st.session_state.button_clicked = False
    with col[0]:
        st.write("_Please upload a CSV file to continue._")
else:
    st.session_state.button_clicked = True

    with col[0]:
        df = load_data(uploaded_file)
        config["VARIABLES"] = config_dict
        config["DATAFRAME"] = {"df":df.to_csv(index=False)}
        uploaded_file = None
        # st.dataframe(df)

        save(config)
        # df.columns

#df = load_data('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')

        df.head(5)
            # Observe changes in the target dropdown
#config_dict["target_dropdown"].observe(exclude_target_from_features, names='value')
        st.divider()
        #update_dropdowns(df)
        # Get list of columns from the dataframe
        columns = df.columns.tolist()
            # Update the dropdown options
        cat = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if len(cat) == 0:
            cat = df.select_dtypes(include=["object", "category", 'bool', "number"]).columns.tolist()
        default = cat.index(cat[-1])
        config_dict["target_dropdown"] = st.selectbox(
            "Choose target column: ",
            cat, index=default)
        st.write("You selected:", config_dict["target_dropdown"])
        
    
        # Optionally, pre-select the last column as the target (common convention)
        #config_dict["target_dropdown"].value = columns[-1]
        #target = config_dict["target_dropdown"].value

        X, y = df.drop(columns=config_dict["target_dropdown"]), df[config_dict["target_dropdown"]]
        X.head(5)
        display(config_dict["target_dropdown"])
        orig_classes = df[config_dict["target_dropdown"]].unique()
    #display(config_dict["target_dropdown"])

    # Import label encoder 
        from sklearn import preprocessing 
        from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

    # Step 1: Automatically find the categorical and numerical columns
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        numerical_columns = X.select_dtypes(include=['number', 'bool']).columns

        #st.write(categorical_columns)
        #st.write(numerical_columns)

    # Step 2: Set up the column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(), categorical_columns),  # One-hot encode categorical columns
                ('scaler', MinMaxScaler(), numerical_columns)  # Apply MinMaxScaler to numerical columns
            ]
        )


    #LABEL ENCODING
    # label_encoder object knows  
    # how to understand word labels. 
        label_encoder = preprocessing.LabelEncoder()  
    # Encode labels in column 'species'. 
        df[config_dict["target_dropdown"]]= label_encoder.fit_transform(df[config_dict["target_dropdown"]]) 


        columns = df.columns.tolist()
        feature_options = [col for col in columns if col != config_dict["target_dropdown"]]

        features = feature_options   

        print(features)
        print(config_dict["target_dropdown"])

        X = df[features]  # Features
        y = df[config_dict["target_dropdown"]]      # Target

        X.head(5)
        y.head(5)

        #Scatterplot Graph - Analysis
        y_orig = label_encoder.inverse_transform(df[config_dict["target_dropdown"]])
        x_axis = st.selectbox("Choose x axis: ", X.columns.tolist())
        y_axis = st.selectbox("Choose y axis", X.columns.tolist())
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=y_orig);
        ax.set_title("Feature Scatterplot Analysis")
        st.write(fig)

        from ipywidgets import Layout
        style = {'description_width': 'initial'}

        config_dict["max_depth_slider"] = st.slider(value=int(config_dict["max_depth_slider"]), min_value=1, max_value=10, step=1, label="Max Depth: ")
    #config_dict["max_depth_slider"] = widgets.IntSlider(value=3, min=1, max=10, step=1, description='Max Depth:', style=style)
    #config_dict["criterion_dropdown"] = widgets.Dropdown(options=['gini', 'entropy', 'log_loss'], value='gini', description='Criterion:')
        if(config_dict["max_depth_slider"] >=5):
            label = 'WARNING: High max_depth levels may cause overfitting!'
            st.write(f":red[**{label}**]")
        if(config_dict["max_depth_slider"] <= 1):
            label = 'WARNING: Low max_depth levels may cause underfitting!'
            st.write(f":red[**{label}**]")

        config_dict["criterion_dropdown"] = st.selectbox("Choose one from the criteria below",
            ("gini", "entropy", "log_loss"),
            index=0,
            placeholder="Select a method...",)

    #config_dict["splitter_dropdown"] = widgets.Dropdown(options=['best', 'random'], value='best', description='Splitter:')
        config_dict["splitter_dropdown"] = st.selectbox("Choose one from the options below",
            ("best", "random"),
            index=0,
            placeholder="Select a method...",)

    #config_dict["min_samples_split_slider"] = widgets.IntSlider(value=2, min=1, max=100, step=1, description="Min Samples Split:", style=style)
        config_dict["min_samples_split_slider"] = st.slider(value=config_dict["min_samples_split_slider"], min_value=1, max_value=100, step=1, label="Min Samples Split: ")
        if(config_dict["min_samples_split_slider"] <=10):
            label = 'WARNING: Low values for the mimimum samples per split may cause overfitting!'
            st.write(f":red[**{label}**]")
    #config_dict["min_samples_leaf_slider"] = widgets.IntSlider(value=1, min=1, max=50, step=1, description="Min Samples Leaf:",style=style)
        config_dict["min_samples_leaf_slider"] = st.slider(value=config_dict["min_samples_leaf_slider"], min_value=1, max_value=50, step=1, label="Min Samples Leaf: ")
        if(config_dict["min_samples_leaf_slider"] <=10):
            label = 'WARNING: Low values for the mimimum samples per leaf may cause overfitting!'
            st.write(f":red[**{label}**]")


        config_dict["test_split_slider"] = st.slider(value=config_dict["test_split_slider"], min_value=0.05, max_value=1.0, step=0.05, label="Test size: ")
        if(config_dict["test_split_slider"] <0.1):
            label = 'WARNING: A small test split size will not provide a good generalisation of the model!'
            st.write(f":red[**{label}**]")
        elif(config_dict["test_split_slider"] > 0.3):
            label = 'WARNING: A large test split size means there may be not enough data in training to provide a good generalised model!'
            st.write(f":red[**{label}**]")
        st.divider()
        config["VARIABLES"] = config_dict
        save(config)


        cat = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        num = X.select_dtypes(include=['number']).columns.tolist()

        scaler=MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X[num]), columns=X[num].columns)

        #Initialize OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)

        # Apply one-hot encoding to the categorical columns
        one_hot_encoded = encoder.fit_transform(X[cat])

        #Create a DataFrame with the one-hot encoded columns
        #We use get_feature_names_out() to get the column names for the encoded data
        one_hot_X = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(cat))

            # Concatenate the one-hot encoded dataframe with the original dataframe
        X = pd.concat([X_scaled, one_hot_X], axis=1)

        # Display the resulting dataframe
        st.dataframe(X)



        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=config_dict["test_split_slider"], random_state=20231020)
        #CHOOSE TEST AND TRAIN SIZE, RANDOM_STATE

        
    # Step 5: Create a pipeline with the column transformer and classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(max_depth=config_dict["max_depth_slider"], criterion=config_dict["criterion_dropdown"], splitter=config_dict["splitter_dropdown"], random_state=0))
        ])


    # Step 6: Fit the model
        pipeline.fit(X_train, y_train)

        

        st.subheader("Feature Selection")
        ref_selector = RFECV(pipeline['classifier'], step=1, cv=5)
        ref_selector.fit(X, y)
        X_sel = pd.DataFrame(ref_selector.transform(X))
        #st.write(X_sel.head())
        #st.write(ref_selector.get_feature_names_out())
        n_scores = len(ref_selector.cv_results_["mean_test_score"])
        #n_scores

        try:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.errorbar(
            range(1, n_scores+1),
            ref_selector.cv_results_["mean_test_score"],
            yerr=ref_selector.cv_results_["std_test_score"],
            )
            ax.set_xlabel("Number of features selected")
            ax.set_ylabel("Mean test score");
            st.write(fig)
        except AttributeError as e:
            st.error(f"Plot rendering failed: {e}")

        n_features = st.slider('Number of features to select', min_value=1, max_value=X_train.shape[1], value=1)
        

        # Create RFE model and fit it
        rfe = RFE(pipeline['classifier'], n_features_to_select=n_features)
        rfe.fit(X_train, y_train)

        # Show selected features
        st.write("Selected Features: ", X_train.columns[rfe.support_])

        if st.button("Retrain classifier with selected features"):
            # Use the selected features to retrain the classifier
            X_train_selected = pd.DataFrame(rfe.transform(X_train))
            X_test_selected = rfe.transform(X_test)

            # Retrain the classifier on the selected features
            classifier_retrained = DecisionTreeClassifier(max_depth=config_dict["max_depth_slider"], criterion=config_dict["criterion_dropdown"], splitter=config_dict["splitter_dropdown"], random_state=0)  # Same parameters as before
            classifier_retrained.fit(X_train_selected, y_train)

            # Display the retrained classifier results
            st.write("Classifier retrained on selected features")
            training_accuracy = balanced_accuracy_score(classifier_retrained.predict(X_train_selected), y_train)
            test_accuracy = balanced_accuracy_score(classifier_retrained.predict(X_test_selected), y_test)
            st.write(f"Training score: {training_accuracy}")
            st.write(f"Testing score: {test_accuracy}")
            st.write(f"Generalisation score: {training_accuracy - test_accuracy}")
        
            
            X_selected = X.loc[:, rfe.support_]
            selected_features = X.columns[rfe.support_]
            st.dataframe(X_selected)

            dtviz_selected = dtreeviz.model(
                classifier_retrained, X_selected, y, target_name=config_dict["target_dropdown"], feature_names=selected_features,
                class_names=orig_classes)
            
            svg=dtviz_selected.view()._repr_svg_()
            st.markdown(
                """
                <style>
                    button[title^=Exit]+div [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """, unsafe_allow_html=True
            )
            st.markdown(
                """
                <style>
                    [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """, unsafe_allow_html=True
            )

            try:
                st.image(svg)
            except AttributeError as e:
                st.error(f"Plot rendering failed: {e}")


    with col[1]:
        st.subheader("Decision Tree Graph and Results", divider='grey')
        dt = DecisionTreeClassifier(max_depth=config_dict["max_depth_slider"], criterion=config_dict["criterion_dropdown"], splitter=config_dict["splitter_dropdown"])
        dt.fit(X_train, y_train)

        dtviz = dtreeviz.model(
            pipeline['classifier'], preprocessor.fit_transform(X), y, target_name=config_dict["target_dropdown"], feature_names=features,
            class_names=orig_classes)


    # dtviz = dtreeviz.model(dt, X, y, target_name=config_dict["target_dropdown"], feature_names=features, class_names=orig_classes)

        svg=dtviz.view()._repr_svg_()
        st.markdown(
                """
                <style>
                    button[title^=Exit]+div [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """, unsafe_allow_html=True
            )
        st.markdown(
                """
                <style>
                    [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """, unsafe_allow_html=True
            )

        try:
            st.image(svg)
        except AttributeError as e:
            st.error(f"Plot rendering failed: {e}")

        st.divider()

        training_score = balanced_accuracy_score(y_train, pipeline.predict(X_train))
        test_score = balanced_accuracy_score(y_test, pipeline.predict(X_test))
        generalisation_score = training_score - test_score

        st.subheader("Scores _(balanced accuracy)_")
        st.write("Training score: ", training_score)
        st.write("Test score: ", test_score)
        st.write("Generalisation score: ", generalisation_score)

        if(generalisation_score > 0.1):
            label = "WARNING: The scores suggest that the model may be overfitting!"
            st.write(f"**:red[{label}]**")
        elif(training_score < 0.7 and test_score < 0.7):
            label = "WARNING: The scores suggest that the model may be underfitting!"
            st.write(f"**:red[{label}]**")
        elif(generalisation_score<0.05 and training_score > 0.85):
            label = "The scores suggest this model is a good fit"
            st.write(f"**:red[{label}]**")
        else:
            label = "WARNING: The model may need some improving"
            st.write(f"**:red[{label}]**")


       

        st.divider()

        y_pred = pipeline.predict(X_test)
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_test_decoded, y_pred_decoded)
        st.subheader("Confusion Matrix")

        try:
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
            st.write(label_encoder.inverse_transform(pipeline['classifier'].classes_))
            ax.xaxis.set_ticklabels(label_encoder.inverse_transform(pipeline['classifier'].classes_)); ax.yaxis.set_ticklabels(label_encoder.inverse_transform(pipeline['classifier'].classes_));
            st.pyplot(fig)
        except AttributeError as e:
            st.error(f"Plot rendering failed: {e}")


        classes=pipeline.classes_

        st.divider()

        import graphviz
        display("HERE")
        graphviz.Source(dtviz.view().dot)

        display("END")
        dtviz.view(scale=1.5, fontname='sans-serif', orientation="LR")

        # x = df[features].iloc[10]
        # dtviz.view(x=x, scale=1.5, fontname='sans-serif')

        # dtviz.view(x=x, show_just_path=True) 

        # print(dtviz.explain_prediction_path(x))

        # print(export_text(pipeline['classifier'], feature_names=X.columns, class_names=orig_classes))

        # dtviz.instance_feature_importance(x, figsize=(3.5,2))

        # dtviz.leaf_sizes(figsize=(3.5,2))

        # dtviz.ctree_leaf_distributions(figsize=(3.5,2))

        # dtviz.node_stats(node_id=6)

        # dtviz.leaf_purity(figsize=(3.5,2))


    #CROSS VALIDATION 
        
        st.subheader("Cross Validation for Model Evaluation")
        config_dict["cross_val"] = st.slider(value=config_dict["cross_val"], min_value=2, max_value=10, step=1, label='CV (Folds):')
        scoring_slider = st.selectbox("Choose a Scoring Type: ", ('accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'f1',
                                            'f1_micro', 'f1_macro', 'f1_weighted', 'neg_log_loss', 'precision',
                                            'precision_micro', 'precision_macro', 'precision_weighted',
                                            'recall', 'recall_micro', 'recall_macro', 'recall_weighted',
                                            'jaccard', 'jaccard_micro', 'jaccard_macro', 'jaccard_weighted',
                                            'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'), index=config_dict["scoring_index"])
        display(config_dict["cross_val"], scoring_slider)

        scores = cross_val_score(pipeline, X, y, cv=config_dict["cross_val"], scoring=scoring_slider)

        st.write("Cross Validation Score (", scoring_slider,"): ", scores)
        st.write("Scores _(mean)_: ", scores.mean())
        st.write("Scores _(standard deviation)_: ", scores.std())

        st.divider()

        #FEATURE IMPORTANCE
        feature_importances = pipeline['classifier'].feature_importances_
        feature_labels = X_test.columns
        sorted_idx = np.argsort(feature_importances)[::-1]

        feature_importance_df = pd.DataFrame({
            'Feature': feature_labels[sorted_idx],
            'Importance': feature_importances[sorted_idx]
        })

        st.subheader("Feature Importance")
        st.dataframe(feature_importance_df)

        try:
            fig = plt.figure(figsize=(12, 6))
            plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx][::-1], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx][::-1])
            plt.ylabel("Features", fontsize='14')
            plt.xlabel("Importance", fontsize='14')
            plt.title("Feature Importance Plot", fontsize='20')
            st.pyplot(fig)
        except AttributeError as e:
            st.error(f"Plot rendering failed: {e}")

        st.divider()

        #COST COMPLEXITY PRUNING
        st.subheader("Cost Complexity Pruning")
        path = pipeline["classifier"].cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities

        ccp_df = pd.DataFrame({
            'CCP_Alphas': ccp_alphas,
            'Impurities': impurities
        })
        st.dataframe(ccp_df)


        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set")
        st.write(fig)

        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha, max_depth=config_dict["max_depth_slider"], criterion=config_dict["criterion_dropdown"], splitter=config_dict["splitter_dropdown"])
            clf.fit(X_train, y_train)
            clfs.append(clf)
        st.write(
            "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
                clfs[-1].tree_.node_count, ccp_alphas[-1]
            )
        )

        clfs = clfs[:-1]
        ccp_alphas = ccp_alphas[:-1]

        node_counts = [clf.tree_.node_count for clf in clfs]
        depth = [clf.tree_.max_depth for clf in clfs]

        try:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
            ax[0].set_xlabel("alpha")
            ax[0].set_ylabel("number of nodes")
            ax[0].set_title("Number of nodes vs alpha")
            ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
            ax[1].set_xlabel("alpha")
            ax[1].set_ylabel("depth of tree")
            ax[1].set_title("Depth vs alpha")
            fig.tight_layout()
            st.write(fig)
        except AttributeError as e:
            st.error(f"Plot rendering failed: {e}")

        train_scores = [clf.score(X_train, y_train) for clf in clfs]
        test_scores = [clf.score(X_test, y_test) for clf in clfs]

        try:
            fig, ax = plt.subplots()
            ax.set_xlabel("alpha")
            ax.set_ylabel("accuracy")
            ax.set_title("Accuracy vs alpha for training and testing sets")
            ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
            ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
            ax.legend()
            st.write(fig)
        except AttributeError as e:
            st.error(f"Plot rendering failed: {e}")

        # Choose the best pruned tree based on test set performance
        test_scores = [clf.score(X_test, y_test) for clf in clfs]
        best_tree_index = np.argmax(test_scores)
        best_clf = clfs[best_tree_index]

        st.write(f"Best alpha: {ccp_alphas[best_tree_index]}, Test Accuracy: {test_scores[best_tree_index]}")

        clf_model = dtreeviz.model(
            best_clf, X, y, target_name=config_dict["target_dropdown"], feature_names=features,
            class_names=orig_classes)
        
        svg=clf_model.view()._repr_svg_()
        st.markdown(
                """
                <style>
                    button[title^=Exit]+div [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """, unsafe_allow_html=True
            )
        st.markdown(
                """
                <style>
                    [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """, unsafe_allow_html=True
            )
        try:
            st.image(svg)
        except AttributeError as e:
            st.error(f"Plot rendering failed: {e}")
        
        training_score_2 = balanced_accuracy_score(y_train, best_clf.predict(X_train))
        test_score_2 = balanced_accuracy_score(y_test, best_clf.predict(X_test))
        generalisation_score_2 = training_score_2 - test_score_2

        st.subheader("Scores _(balanced accuracy)_")
        st.write("Training score: ", training_score_2)
        st.write("Test score: ", test_score_2)
        st.write("Generalisation score: ", generalisation_score_2)

        if(generalisation_score_2 > 0.1):
            label = "WARNING: The scores suggest that the model may be overfitting!"
            st.write(f"**:red[{label}]**")
        elif(training_score_2 < 0.7 and test_score_2 < 0.7):
            label = "WARNING: The scores suggest that the model may be underfitting!"
            st.write(f"**:red[{label}]**")
        elif(generalisation_score_2<0.05 and training_score_2 > 0.85):
            label = "The scores suggest this model is a good fit"
            st.write(f"**:red[{label}]**")
        else:
            label = "WARNING: The model may need some improving"
            st.write(f"**:red[{label}]**")


        
        #SHAP 

            #from streamlit_shap import st_shap
            # compute SHAP values
            #st.subheader("SHAP Analysis")
            #explainer = shap.TreeExplainer(pipeline['classifier'])
            #shap_values = explainer.shap_values(X)

            #explainer = shap.Explainer(pipeline['classifier'])
            #shap_values = explainer(X)

            #shap_values.shape
            #156: The number of data points (samples) you are analyzing.
            #4: The number of features in your model.
            #3: The number of classes your model is predicting. SHAP provides one set of feature contributions per class.
            #num_classes = len(pipeline['classifier'].classes_) 
            #st.write(num_classes)
            #class_0 = orig_classes[0]
            #st.write(X.columns)

            
            #for i in range(num_classes):
            #    st.write("Summary plot for class ", orig_classes[i])
            #    st_shap(shap.summary_plot(shap_values[:,:,i], X, feature_names = columns))

            #for i in range(num_classes):
            #    st.write("Beeswarm plots for class", orig_classes[i])
                
            #st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], X))
            #st_shap(shap.summary_plot(shap_values, X, plot_type="bar", class_names= classes, feature_names = X.columns))

            
            #st.write("Class 0")
            #st_shap(shap.plots.beeswarm(shap_values[:,:,0]), height=300)

            #st.write("Class 1")
            #st_shap(shap.plots.beeswarm(shap_values[:,:,1]), height=300)

            #st.write("Class 2")
            #st_shap(shap.plots.beeswarm(shap_values[:,:,2]), height=300)

            #st.write("Summary Plot for", class_0)
            #st_shap(shap.summary_plot(shap_values[:,:,0], X, feature_names = X.columns))

            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)


    with col[2]:
        #GRID SEARCH
        #Perform a grid search

        st.subheader("Grid Search", divider='grey')
        
        config_dict["max_depth_inputs2"] = st_tags(
            label="Enter inputs for max depth elements:",
            text='Press enter to add more',
            value=config_dict["max_depth_inputs2"],
            suggestions=['1', '3', '5'],
            maxtags = 10,
            key='1')
        
        
            
        # st.write(np.asarray(config_dict["max_depth_inputs2"]).dtype)
        
        bad_array = [x for x in np.asarray(config_dict["max_depth_inputs2"]) if not x.isdigit()]
        if(len(bad_array) > 0): st.error(str(bad_array) + " are invalid datatypes.")
        max_depth_array = np.array([x for x in np.asarray(config_dict["max_depth_inputs2"]) if x.isdigit()],dtype='int')
        # max_depth_array = max_depth_array[max_depth_array >= 0]
        # max_depth_array = np.asarray(config_dict["max_depth_inputs2"], dtype='int')
        #st.write(max_depth_array)

        
        st.write("Chosen max_depth elements: ", max_depth_array)

        config_dict["min_samples_inputs2"] = st_tags(
            label="Enter inputs for min samples:",
            text='Press enter to add more',
            value=config_dict["min_samples_inputs2"],
            suggestions=['10', '20', '30'],
            maxtags = 10,
            key='2')
        
        bad_array_samples = [x for x in np.asarray(config_dict["min_samples_inputs2"]) if not x.isdigit()]
        if(len(bad_array_samples) > 0): st.error(str(bad_array_samples) + " are invalid datatypes.")
        min_samples_array = np.array([x for x in np.asarray(config_dict["min_samples_inputs2"]) if x.isdigit()],dtype='int')
        # max_depth_array = max_depth_array[max_depth_array >= 0]
        # max_depth_array = np.asarray(config_dict["max_depth_inputs2"], dtype='int')
        #st.write(min_samples_array)

        # min_samples_array = np.asarray(config_dict["min_samples_inputs2"], dtype='int')

        st.write("Chosen min_sample elements: ", min_samples_array)

        config["VARIABLES"] = config_dict
        save(config)

        b = time.time()
        print("Load time: ",b-a)

        if st.button("Perform grid search"):
            param_grid = {
            'max_depth': max_depth_array,
            'min_samples_leaf':min_samples_array}
        
            from sklearn.model_selection import ParameterGrid
        # this is the Parameter Grid of the above: all possible combinations of the values
        # of the two hyper-parameters (4 max_depth x 3 min_sample_leaf = 12 configuration)
            list(ParameterGrid(param_grid))

            from sklearn.model_selection import GridSearchCV
            clf = GridSearchCV(pipeline['classifier'], param_grid, return_train_score=True)

        # the results from a grid search produces some interesting properties
        # the `cv_results_` is a fitted parameter (a dict) that can be made into
        # a DataFrame for easier analysis of the configurations and their performance
            gs_results = clf.fit(X_train, y_train)

            gs_df = pd.DataFrame(gs_results.cv_results_)
        
            st.subheader("Grid Search CV Results ")
            st.dataframe(gs_df[ [
            'param_max_depth', 'param_min_samples_leaf',
            'mean_train_score', 'std_train_score',
            'mean_test_score', 'std_test_score', 'rank_test_score'
            ] ].sort_values('rank_test_score'))

            st.divider()

            st.subheader("Best Model Parameters")
            gs_results.best_params_
            #gs_results.best_estimator_
            best_model = gs_results.best_estimator_

            st.divider()

            st.subheader("Best Model Scores _(balanced accuracy)_")
            st.write("Best Model Training Score: ", balanced_accuracy_score(y_train, best_model.predict(X_train)))
            st.write("Best Model Test Score: ", balanced_accuracy_score(y_test, best_model.predict(X_test)))

            st.divider()

            st.subheader("Best Model Decision Tree")
            dtviz = dtreeviz.model(
            best_model, X, y, target_name=config_dict["target_dropdown"], feature_names=features,
            class_names=orig_classes)

            svg=dtviz.view()._repr_svg_()
            st.markdown(
                """
                <style>
                    button[title^=Exit]+div [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """, unsafe_allow_html=True
            )
            st.markdown(
                """
                <style>
                    [data-testid=stImage]{
                        text-align: center;
                        display: block;
                        margin-left: auto;
                        margin-right: auto;
                        width: 100%;
                    }
                </style>
                """, unsafe_allow_html=True
            )

            try:
                st.image(svg)
            except AttributeError as e:
                st.error(f"Plot rendering failed: {e}")
            # st.write(config_dict)
            

            # creating final config dictionary

            # opening ini file (w means create new if not aval)

            # accessing config




