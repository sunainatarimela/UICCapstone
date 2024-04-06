import streamlit as st
import streamlit.components.v1 as components


import pandas as pd
import numpy as np
import pickle
import requests, os
import base64

apptitle = 'Gov Contracts'
st.set_page_config(page_title=apptitle, page_icon=":book:")

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: linear-gradient(rgba(255,255,255,0.85), rgba(255,255,255,0.85)), url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background("govcontracts.jpg")

with open("uicbusiness.png", "rb") as f:
    data = base64.b64encode(f.read()).decode("utf-8")

    st.markdown(
        f"""
        <div style="margin-top:-.5%;margin-left:-5%;">
            <img src="data:image/png;base64,{data}" width="200" height="100">
        </div>
        """,
        unsafe_allow_html=True,
    )

def intro():
    import streamlit as st

    st.write("# Welcome Government contract data prediction during a pandemic")
    st.sidebar.success("Select an option above.")

    st.markdown(
        """
        **Welcome!

        Government Contracts are contracts that are given to various vendors to complete the task at hand. There are some contracts that are
        released everyday depending on the need of the related government agency.

        One of the websites to get updated government contracts related information is [sam.gov](https://sam.gov/content/home)

        If you are a vendor for government contracts and want to see details about the contract assignments, then you are at the right place!

        **👈 Select an option from the dropdown on the left to see the business type that would win a contract **
    """
    )
                                                              # """THIS ONE WORKS"""
def business_type_predict():
    import pandas as pd
    import numpy as np
    import pickle
    import requests, os
    import base64
    import streamlit as st
    import streamlit.components.v1 as components


    def construct_sample(input):

        # Load the serialized object from the pickle file
        with open('label_encoder.pkl', 'rb') as file:
          label_encoders = pickle.load(file)

        X_test = np.zeros(20)
        X_test[0] = label_encoders['Contracting Agency ID'].transform([input[0]])
        X_test[1] = label_encoders['Domestic or Foreign Entity'].transform([input[5]])
        X_test[2] = label_encoders['Is Performance Based Service Acquisition'].transform([input[6]])
        X_test[3] = label_encoders['Type of Contract'].transform([input[1]])
        X_test[4] = label_encoders['NAICS Code'].transform([input[2][:2]])
        X_test[5] = label_encoders['Principal Place of Performance State Code'].transform([input[3]])
        X_test[6] = label_encoders['Principal Place of Performance Country Name'].transform([input[4]])
        X_test[7] = label_encoders['Country of Product or Service Origin'].transform([input[19]])
        X_test[8] = label_encoders['Extent Competed'].transform([input[7]])
        X_test[9] = label_encoders['Solicitation Procedures'].transform([input[8]])
        X_test[10] = label_encoders['Local Area Set Aside'].transform([input[9]])
        X_test[11] = label_encoders['Vendor Address State Name'].transform([input[10]])
        X_test[12] = label_encoders['Vendor Address Country Name'].transform([input[11]])
        X_test[13] = label_encoders['Labor Standards'].transform([input[12]])
        X_test[14] = label_encoders['Is Vendor Business Type - For Profit Organization'].transform([input[13]])
        X_test[15] = label_encoders['Is Vendor Business Type - All Awards'].transform([input[14]])
        X_test[16] = label_encoders['Is Vendor Business Type - Corporate Entity, Not Tax Exempt'].transform([input[15]])
        X_test[17] = label_encoders['Is Vendor Business Type - Manufacturer Of Goods'].transform([input[16]])
        X_test[18] = label_encoders['Base and All Options Value (Total Contract Value)'].transform(np.array([input[17]]).reshape(-1, 1))
        X_test[19] = label_encoders['Duration of Contract'].transform(np.array([input[18]]).reshape(-1, 1))

        return X_test

    def run_pred_model_business_type (select_agencyid : str,select_contracttype : str,select_naicscode : str,select_pricipalplaceofperformancestate : str,select_pricipalplaceofperformancecountry: str,select_entity: str,select_performancebasedservice: str,select_extentcompeted: str,select_solicitationprocedures: str,select_localareasetaside: str,select_vendoraddresstatename: str,select_vendoraddresscountryname: str,select_laborstandards: str,select_vendorbusinesstypeforProfit: str,select_vendorbusinesstypeallawards: str,select_vendorbusinesstypecorprateentity: str,select_vendorbusinesstypeManufactofgoods: str,select_contractvalue: str,select_contractduration: str,select_countryofprodservorigin: str):


        #Generate prediction
        input = [select_agencyid,select_contracttype,select_naicscode,select_pricipalplaceofperformancestate,select_pricipalplaceofperformancecountry,
                        select_entity,select_performancebasedservice,select_extentcompeted,select_solicitationprocedures,select_localareasetaside,
                        select_vendoraddresstatename,select_vendoraddresscountryname,select_laborstandards,select_vendorbusinesstypeforProfit,
                        select_vendorbusinesstypeallawards,select_vendorbusinesstypecorprateentity,select_vendorbusinesstypeManufactofgoods,
                        select_contractvalue,select_contractduration, select_countryofprodservorigin]

        X_test = construct_sample(input)

        # Load the serialized object from the pickle file
        with open('clf.pkl', 'rb') as file:
          loaded_model = pickle.load(file)

        prediction = loaded_model.predict(X_test.reshape(1, -1) )

        #Predicted Business Type
        #BusinessType = label_encoders['Business Type'].inverse_transform([prediction])
        #return BusinessType

        #Predict Probabilities
        pred_probabilities = loaded_model.predict_proba(X_test.reshape(1, -1))


        # Load the serialized object from the pickle file
        with open('label_encoder.pkl', 'rb') as file:
          label_encoders = pickle.load(file)

        prob_df=pd.DataFrame(pred_probabilities, columns=label_encoders['Business Type'].classes_)
        return prob_df

    st.write("# Predicting the Business Type that will win the contract")
    st.markdown(
        """
        ** You are now ready to predict which Business Type will win the contract.**
            Please fill out the fields on the left and click on the button below to see the output.
            You can also see the latest trands by clicking on the tableau dashboard link provided.

        """
        )



    #-- Set business type

    select_agencyid = st.sidebar.text_input(label="Contracting Agency ID", placeholder="1406")

    select_contracttype = st.sidebar.selectbox('Contract Type',
                                        ['FIRM FIXED PRICE','TIME AND MATERIALS','LABOR HOURS','FIXED PRICE AWARD FEE','FIXED PRICE WITH ECONOMIC PRICE ADJUSTMENT','COST NO FEE','COST PLUS FIXED FEE','COST PLUS AWARD FEE','COST PLUS INCENTIVE FEE','FIXED PRICE INCENTIVE','FIXED PRICE LEVEL OF EFFORT','COST SHARING','FIXED PRICE REDETERMINATION'])

    select_naicscode = st.sidebar.text_input(label="NAICS Code", placeholder="622109")

    select_pricipalplaceofperformancestate = st.sidebar.selectbox('Principal place of performance state code', ['VA','DC','CA','MD','IN','FL','NY','MO','CO','AK','LA','ID','WY','SD','MT','OR','WA','VT','PA','NM','NJ','IL','MN','TX','WV','AL','PR','KY','SC','NC','OK','GA','AR','MS','NE','MI','OH','IA','DE','NH','KS','WI','AZ','TN','CT','MA','HI','UT','RI','ME','ND','NV','AS','UGANDA','GU','FRANCE','VI','VIETNAM','CANADA','JAPAN','INDIA','CANADA','MEXICO'])

    select_pricipalplaceofperformancecountry = st.sidebar.selectbox('Principal place of performance country name',
                                        ['UNITED STATES','UGANDA','FRANCE','VIETNAM','CANADA','JAPAN','UNITED KINGDOM','BAHRAIN','INDIA','CANADA','MEXICO'])

    select_entity = st.sidebar.selectbox("Domestic or Foreign Entity",['U.S. OWNED BUSINESS','OTHER U.S. ENTITY (E.G. GOVERNMENT)','FOREIGN-OWNED BUSINESS INCORPORATED IN THE U.S.','FOREIGN-OWNED BUSINESS NOT INCORPORATED IN THE U.S.','OTHER FOREIGN ENTITY (E.G. FOREIGN GOVERNMENT)'])

    select_performancebasedservice = st.sidebar.radio("Performance Based Service Acquisition",('NO - SERVICE WHERE PBA IS NOT USED.','YES - SERVICE WHERE PBA IS USED.','NOT APPLICABLE'))

    select_extentcompeted = st.sidebar.selectbox('Extent Competed',
                                        ['FULL AND OPEN COMPETITION','NOT COMPETED UNDER SAP','FULL AND OPEN COMPETITION AFTER EXCLUSION OF SOURCES','NOT COMPETED','COMPETED UNDER SAP','NOT AVAILABLE FOR COMPETITION'])

    select_solicitationprocedures = st.sidebar.selectbox('Solicitation Procedures',
                                        ['SUBJECT TO MULTIPLE AWARD FAIR OPPORTUNITY','SIMPLIFIED ACQUISITION','NEGOTIATED PROPOSAL/QUOTE','ONLY ONE SOURCE','ALTERNATIVE SOURCES','SEALED BID','ARCHITECT-ENGINEER FAR 6.102','BASIC RESEARCH','TWO STEP'])

    select_localareasetaside = st.sidebar.radio("Local Area Set Aside",('YES','NO'))

    select_vendoraddresstatename = st.sidebar.selectbox('Vendor Address State',
                                        ['ILLINOIS','ARIZONA','TEXAS','CALIFORNIA','NEW YORK','NEW JERSEY','NORTH CAROLINA'])

    select_vendoraddresscountryname = st.sidebar.selectbox('Vendor address country name',
                                        ['UNITED STATES','INDIA','CANADA','MEXICO'])

    select_laborstandards = st.sidebar.radio("Labor Standards",('YES','NO','NOT APPLICABLE'))

    select_vendorbusinesstypeforProfit = st.sidebar.radio("Is Vendor Business type - For Profit Organisation",('YES','NO'))

    select_vendorbusinesstypeallawards = st.sidebar.radio("Is Vendor Business type - All Awards ",('YES','NO'))

    select_vendorbusinesstypecorprateentity = st.sidebar.radio("Is Vendor Business type - Corporate Entity, Not tax Exempt",('YES','NO'))

    select_vendorbusinesstypeManufactofgoods = st.sidebar.radio("Is Vendor Business type - Manufacturer of Goods",('YES','NO'))

    select_contractvalue = st.sidebar.text_input(label="Total Contract Value", placeholder="$25,000,000")

    select_contractduration = st.sidebar.text_input(label="Contract Duration(in days)", placeholder="365")

    select_countryofprodservorigin = st.sidebar.selectbox('Country of Product or Service Origin',
                                        ['UNITED STATES','INDIA','CANADA','MEXICO'])


    html_temp = "<div class='tableauPlaceholder' id='viz1711381846728' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IDS_560_dashboard&#47;Dashboard1' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1711381846728');                    var vizElement = divElement.getElementsByTagName('object')[0];                 if ( divElement.offsetWidth > 800 ) { vizElement.style.width='2850px';vizElement.style.height='1727px';}else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='2850px';vizElement.style.height='1727px';} else { vizElement.style.width='100%';vizElement.style.height='2077px';}               var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
    components.html(html_temp)

    def main():
      import pickle
      # Load the serialized object from the pickle file
      with open('clf.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

      # # Load the serialized object from the pickle file
      with open('label_encoder.pkl', 'rb') as file:
        label_encoders = pickle.load(file)



    html_temp = "<div class='tableauPlaceholder' id='viz1710733360364' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IDS_560_dashboard&#47;Dashboard1' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1710733360364');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='1620px';vizElement.style.maxWidth='1720px';vizElement.style.width='100%';vizElement.style.minHeight='818px';vizElement.style.maxHeight='910px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='1620px';vizElement.style.maxWidth='1720px';vizElement.style.width='100%';vizElement.style.minHeight='818px';vizElement.style.maxHeight='910px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1250px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
    components.html(html_temp)

    if st.button ("Predict the business type to win the contract"):
        output = run_pred_model_business_type(select_agencyid,select_contracttype,select_naicscode,select_pricipalplaceofperformancestate,
                    select_pricipalplaceofperformancecountry,select_entity,select_performancebasedservice,select_extentcompeted,select_solicitationprocedures,
                    select_localareasetaside,select_vendoraddresstatename,select_vendoraddresscountryname,select_laborstandards,
                    select_vendorbusinesstypeforProfit,select_vendorbusinesstypeallawards,select_vendorbusinesstypecorprateentity,
                    select_vendorbusinesstypeManufactofgoods,select_contractvalue,select_contractduration,select_countryofprodservorigin)

        st.success('The business that will win the contract is{}'.format(output))
        st.table(output)
    #st.dataframe(output.style.highlight_max(axis=1))

    if __name__ == "__main__":
          main()

def contract_value_predict():
    import pandas as pd
    import numpy as np
    import pickle
    import requests, os
    import base64
    import streamlit as st
    import streamlit.components.v1 as components



    def run_pred_model_business_type (select_businesstype: str,select_agencyid : str,select_contracttype : str,select_naicscode : str,select_pricipalplaceofperformancestate : str,
                    select_pricipalplaceofperformancecountry: str,select_entity: str,select_performancebasedservice: str,select_extentcompeted: str,select_solicitationprocedures: str,
                    select_localareasetaside: str,select_vendoraddresstatename: str,select_vendoraddresscountryname: str,select_laborstandards: str,
                    select_vendorbusinesstypeforProfit: str,select_vendorbusinesstypeallawards: str,select_vendorbusinesstypecorprateentity: str,
                    select_vendorbusinesstypeManufactofgoods: str,select_contractvalue: str,select_contractduration: str):
        """Generate prediction."""



    #-- Set business type

    select_businesstype = st.sidebar.selectbox('What business type do you want to see?',
                                        ['Women Owned','Veteran Owned', 'Small Business'])

    select_agencyid = st.sidebar.text_input(label="Contracting Agency ID", placeholder="1406")

    select_contracttype = st.sidebar.selectbox('Contract Type',
                                        ['FIRM FIXED PRICE','TIME AND MATERIALS','LABOR HOURS','FIXED PRICE AWARD FEE','FIXED PRICE WITH ECONOMIC PRICE ADJUSTMENT','COST NO FEE','COST PLUS FIXED FEE','COST PLUS AWARD FEE','COST PLUS INCENTIVE FEE','FIXED PRICE INCENTIVE','FIXED PRICE LEVEL OF EFFORT','COST SHARING','FIXED PRICE REDETERMINATION'])

    select_naicscode = st.sidebar.text_input(label="NAICS Code", placeholder="622109")

    select_pricipalplaceofperformancestate = st.sidebar.selectbox('Principal place of performance state code', ['VA','DC','CA','MD','IN','FL','NY','MO','CO','AK','LA','ID','WY','SD','MT','OR','WA','VT','PA','NM','NJ','IL','MN','TX','WV','AL','PR','KY','SC','NC','OK','GA','AR','MS','NE','MI','OH','IA','DE','NH','KS','WI','AZ','TN','CT','MA','HI','UT','RI','ME','ND','NV','AS','UGANDA','GU','FRANCE','VI','VIETNAM','CANADA','JAPAN','INDIA','CANADA','MEXICO'])

    select_pricipalplaceofperformancecountry = st.sidebar.selectbox('Principal place of performance country name',
                                        ['UNITED STATES','UGANDA','FRANCE','VIETNAM','CANADA','JAPAN','UNITED KINGDOM','BAHRAIN','INDIA','CANADA','MEXICO'])

    select_entity = st.sidebar.selectbox("Domestic or Foreign Entity",['U.S. OWNED BUSINESS','OTHER U.S. ENTITY (E.G. GOVERNMENT)','FOREIGN-OWNED BUSINESS INCORPORATED IN THE U.S.','FOREIGN-OWNED BUSINESS NOT INCORPORATED IN THE U.S.','OTHER FOREIGN ENTITY (E.G. FOREIGN GOVERNMENT)'])

    select_performancebasedservice = st.sidebar.radio("Performance Based Service Acquisition",('NO - SERVICE WHERE PBA IS NOT USED.','YES - SERVICE WHERE PBA IS USED.','NOT APPLICABLE'))

    select_extentcompeted = st.sidebar.selectbox('Extent Competed',
                                        ['FULL AND OPEN COMPETITION','NOT COMPETED UNDER SAP','FULL AND OPEN COMPETITION AFTER EXCLUSION OF SOURCES','NOT COMPETED','COMPETED UNDER SAP','NOT AVAILABLE FOR COMPETITION'])

    select_solicitationprocedures = st.sidebar.selectbox('Solicitation Procedures',
                                        ['SUBJECT TO MULTIPLE AWARD FAIR OPPORTUNITY','SIMPLIFIED ACQUISITION','NEGOTIATED PROPOSAL/QUOTE','ONLY ONE SOURCE','ALTERNATIVE SOURCES','SEALED BID','ARCHITECT-ENGINEER FAR 6.102','BASIC RESEARCH','TWO STEP'])

    select_localareasetaside = st.sidebar.radio("Local Area Set Aside",('YES','NO'))

    select_vendoraddresstatename = st.sidebar.selectbox('Vendor Address State',
                                        ['ILLINOIS','ARIZONA','TEXAS','CALIFORNIA','NEW YORK','NEW JERSEY','NORTH CAROLINA'])

    select_vendoraddresscountryname = st.sidebar.selectbox('Vendor address country name',
                                        ['UNITED STATES','INDIA','CANADA','MEXICO'])

    select_laborstandards = st.sidebar.radio("Labor Standards",('YES','NO','NOT APPLICABLE'))

    select_vendorbusinesstypeforProfit = st.sidebar.radio("Is Vendor Business type - For Profit Organisation",('YES','NO'))

    select_vendorbusinesstypeallawards = st.sidebar.radio("Is Vendor Business type - All Awards ",('YES','NO'))

    select_vendorbusinesstypecorprateentity = st.sidebar.radio("Is Vendor Business type - Corporate Entity, Not tax Exempt",('YES','NO'))

    select_vendorbusinesstypeManufactofgoods = st.sidebar.radio("Is Vendor Business type - Manufacturer of Goods",('YES','NO'))

    select_contractduration = st.sidebar.text_input(label="Contract Duration(in days)", placeholder="365")

    select_countryofprodservorigin = st.sidebar.selectbox('Country of Product or Service Origin',
                                        ['UNITED STATES','INDIA','CANADA','MEXICO'])
    html_temp = "<div class='tableauPlaceholder' id='viz1711381846728' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IDS_560_dashboard&#47;Dashboard1' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1711381846728');                    var vizElement = divElement.getElementsByTagName('object')[0];                 if ( divElement.offsetWidth > 800 ) { vizElement.style.width='2850px';vizElement.style.height='1727px';}else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='2850px';vizElement.style.height='1727px';} else { vizElement.style.width='100%';vizElement.style.height='2077px';}               var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
    components.html(html_temp)


def contract_duration_predict():
    import pandas as pd
    import numpy as np
    import pickle
    import requests, os
    import base64
    import streamlit as st
    import streamlit.components.v1 as components

        

    def construct_sample_duration(input):

        # Load the serialized object from the pickle file
        with open('label_encoder.pkl', 'rb') as file:
          label_encoders = pickle.load(file)

        X_test = np.zeros(20)
        X_test[0] = label_encoders['Contracting Agency ID'].transform([input[0]])
        X_test[1] = label_encoders['Domestic or Foreign Entity'].transform([input[5]])
        X_test[2] = label_encoders['Is Performance Based Service Acquisition'].transform([input[6]])
        X_test[3] = label_encoders['Type of Contract'].transform([input[1]])
        X_test[4] = label_encoders['NAICS Code'].transform([input[2][:2]])
        X_test[5] = label_encoders['Principal Place of Performance State Code'].transform([input[3]])
        X_test[6] = label_encoders['Principal Place of Performance Country Name'].transform([input[4]])
        X_test[7] = label_encoders['Country of Product or Service Origin'].transform([input[18]])
        X_test[8] = label_encoders['Extent Competed'].transform([input[7]])
        X_test[9] = label_encoders['Solicitation Procedures'].transform([input[8]])
        X_test[10] = label_encoders['Local Area Set Aside'].transform([input[9]])
        X_test[11] = label_encoders['Vendor Address State Name'].transform([input[10]])
        X_test[12] = label_encoders['Vendor Address Country Name'].transform([input[11]])
        X_test[13] = label_encoders['Labor Standards'].transform([input[12]])
        X_test[14] = label_encoders['Is Vendor Business Type - For Profit Organization'].transform([input[13]])
        X_test[15] = label_encoders['Is Vendor Business Type - All Awards'].transform([input[14]])
        X_test[16] = label_encoders['Is Vendor Business Type - Corporate Entity, Not Tax Exempt'].transform([input[15]])
        X_test[17] = label_encoders['Is Vendor Business Type - Manufacturer Of Goods'].transform([input[16]])
        X_test[18] = label_encoders['Business Type'].transform([input[19]])
        X_test[19] = label_encoders['Base and All Options Value (Total Contract Value)'].transform(np.array([input[17]]).reshape(-1, 1))

        return X_test

    def run_pred_duration (select_agencyid : str,select_contracttype : str,select_naicscode : str,select_pricipalplaceofperformancestate : str,
                       select_pricipalplaceofperformancecountry: str,select_entity: str,select_performancebasedservice: str,select_extentcompeted: str,
                       select_solicitationprocedures: str,select_localareasetaside: str,select_vendoraddresstatename: str,select_vendoraddresscountryname: str,
                       select_laborstandards: str,select_vendorbusinesstypeforProfit: str,select_vendorbusinesstypeallawards: str,select_vendorbusinesstypecorprateentity: str,
                       select_vendorbusinesstypeManufactofgoods: str,select_contractvalue: str,select_countryofprodservorigin: str,select_businesstype: str):

          #Generate prediction
      input = [select_agencyid,select_contracttype,select_naicscode,select_pricipalplaceofperformancestate,select_pricipalplaceofperformancecountry,
                          select_entity,select_performancebasedservice,select_extentcompeted,select_solicitationprocedures,select_localareasetaside,
                          select_vendoraddresstatename,select_vendoraddresscountryname,select_laborstandards,select_vendorbusinesstypeforProfit,
                          select_vendorbusinesstypeallawards,select_vendorbusinesstypecorprateentity,select_vendorbusinesstypeManufactofgoods,
                          select_contractvalue,select_countryofprodservorigin,select_businesstype]

      X_test = construct_sample_duration(input)

      # Load the serialized object from the pickle file -> change for duration model
      with open('dur_xgb.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

      # Load the serialized object from the pickle file
      with open('label_encoder.pkl', 'rb') as file:
        label_encoders = pickle.load(file)

      #Regression of the Duration - I think reshaping is not necessary
      prediction = loaded_model.predict(X_test.reshape(1, -1))

      Contract_Duration = label_encoders['Duration of Contract'].inverse_transform([prediction])
      return Contract_Duration

    st.write("# Predicting the Duration of contract")
    st.markdown(
        """
        ** You are now ready to predict which Business Type will win the contract.**
            Please fill out the fields on the left and click on the button below to see the output.
            You can also see the latest trands by clicking on the tableau dashboard link provided.

        """
        )

    #-- Set Duration of contract
    select_businesstype = st.sidebar.selectbox('What business type do you want to see?',
                                    ['Others',	'Is Vendor Business Type - Limited Liability Corporation',	'Is Vendor Business Type - Subchapter S Corporation',
                                     'Is Vendor Business Type - Self-Certified Small Disadvantaged Business',	'Is Vendor Business Type - Women-Owned Business',
                                     'Is Vendor Business Type - The AbilityOne Program',	'Is Vendor Business Type - Partnership or Limited Liability Partnership',
                                     'Is Vendor Business Type - Foreign Owned',	'Is Vendor Business Type - Veteran-Owned Business',
                                     'Is Vendor Business Type - Hispanic American Owned',	'Is Vendor Business Type - Contracts',
                                     'Is Vendor Business Type - 8A Program Participant',	'Is Vendor Business Type - Native American Owned',
                                     'Is Vendor Business Type - Sole Proprietorship',	'Is Vendor Business Type - Subcontinent Asian (Asian-Indian) American Owned',
                                     'Is Vendor Business Type - Black American Owned',	'Is Vendor Business Type - HUBZone Firm',
                                     'Is Vendor Business Type - Asian-Pacific American Owned',
                                     'Is Vendor Business Type - DoT Certified Disadvantaged Business Enterprise',
                                     'Is Vendor Business Type - Other Not For Profit Organization',	'Is Vendor Business Type - Other Minority-Owned'])

    select_agencyid = st.sidebar.text_input(label="Contracting Agency ID", placeholder="1406")

    select_contracttype = st.sidebar.selectbox('Contract Type',
                                        ['FIRM FIXED PRICE','TIME AND MATERIALS','LABOR HOURS','FIXED PRICE AWARD FEE','FIXED PRICE WITH ECONOMIC PRICE ADJUSTMENT','COST NO FEE','COST PLUS FIXED FEE','COST PLUS AWARD FEE','COST PLUS INCENTIVE FEE','FIXED PRICE INCENTIVE','FIXED PRICE LEVEL OF EFFORT','COST SHARING','FIXED PRICE REDETERMINATION'])

    select_naicscode = st.sidebar.text_input(label="NAICS Code", placeholder="622109")

    select_pricipalplaceofperformancestate = st.sidebar.selectbox('Principal place of performance state code', ['VA','DC','CA','MD','IN','FL','NY','MO','CO','AK','LA','ID','WY','SD','MT','OR','WA','VT','PA','NM','NJ','IL','MN','TX','WV','AL','PR','KY','SC','NC','OK','GA','AR','MS','NE','MI','OH','IA','DE','NH','KS','WI','AZ','TN','CT','MA','HI','UT','RI','ME','ND','NV','AS','UGANDA','GU','FRANCE','VI','VIETNAM','CANADA','JAPAN','INDIA','CANADA','MEXICO'])

    select_pricipalplaceofperformancecountry = st.sidebar.selectbox('Principal place of performance country name',
                                        ['UNITED STATES','UGANDA','FRANCE','VIETNAM','CANADA','JAPAN','UNITED KINGDOM','BAHRAIN','INDIA','CANADA','MEXICO'])

    select_entity = st.sidebar.selectbox("Domestic or Foreign Entity",['U.S. OWNED BUSINESS','OTHER U.S. ENTITY (E.G. GOVERNMENT)','FOREIGN-OWNED BUSINESS INCORPORATED IN THE U.S.','FOREIGN-OWNED BUSINESS NOT INCORPORATED IN THE U.S.','OTHER FOREIGN ENTITY (E.G. FOREIGN GOVERNMENT)'])

    select_performancebasedservice = st.sidebar.radio("Performance Based Service Acquisition",('NO - SERVICE WHERE PBA IS NOT USED.','YES - SERVICE WHERE PBA IS USED.','NOT APPLICABLE'))

    select_extentcompeted = st.sidebar.selectbox('Extent Competed',
                                        ['FULL AND OPEN COMPETITION','NOT COMPETED UNDER SAP','FULL AND OPEN COMPETITION AFTER EXCLUSION OF SOURCES','NOT COMPETED','COMPETED UNDER SAP','NOT AVAILABLE FOR COMPETITION'])

    select_solicitationprocedures = st.sidebar.selectbox('Solicitation Procedures',
                                        ['SUBJECT TO MULTIPLE AWARD FAIR OPPORTUNITY','SIMPLIFIED ACQUISITION','NEGOTIATED PROPOSAL/QUOTE','ONLY ONE SOURCE','ALTERNATIVE SOURCES','SEALED BID','ARCHITECT-ENGINEER FAR 6.102','BASIC RESEARCH','TWO STEP'])

    select_localareasetaside = st.sidebar.radio("Local Area Set Aside",('YES','NO'))

    select_vendoraddresstatename = st.sidebar.selectbox('Vendor Address State',
                                        ['ILLINOIS','ARIZONA','TEXAS','CALIFORNIA','NEW YORK','NEW JERSEY','NORTH CAROLINA'])

    select_vendoraddresscountryname = st.sidebar.selectbox('Vendor address country name',
                                        ['UNITED STATES','INDIA','CANADA','MEXICO'])

    select_laborstandards = st.sidebar.radio("Labor Standards",('YES','NO','NOT APPLICABLE'))

    select_vendorbusinesstypeforProfit = st.sidebar.radio("Is Vendor Business type - For Profit Organisation",('YES','NO'))

    select_vendorbusinesstypeallawards = st.sidebar.radio("Is Vendor Business type - All Awards ",('YES','NO'))

    select_vendorbusinesstypecorprateentity = st.sidebar.radio("Is Vendor Business type - Corporate Entity, Not tax Exempt",('YES','NO'))

    select_vendorbusinesstypeManufactofgoods = st.sidebar.radio("Is Vendor Business type - Manufacturer of Goods",('YES','NO'))

    select_contractvalue = st.sidebar.text_input(label="Total Contract Value", placeholder="$25,000,000")

    select_countryofprodservorigin = st.sidebar.selectbox('Country of Product or Service Origin',
                                        ['UNITED STATES','INDIA','CANADA','MEXICO'])

    html_temp1 = "<div class='tableauPlaceholder' id='viz1711381846728' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IDS_560_dashboard&#47;Dashboard1' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1711381846728');                    var vizElement = divElement.getElementsByTagName('object')[0];                 if ( divElement.offsetWidth > 800 ) { vizElement.style.width='2850px';vizElement.style.height='1727px';}else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='2850px';vizElement.style.height='1727px';} else { vizElement.style.width='100%';vizElement.style.height='2077px';}               var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
    components.html(html_temp1)


    def main():
      import pickle
      import pandas as pd
      import os
      import seaborn as sns
      import matplotlib.pyplot as plt
      import numpy as np
      from sklearn import preprocessing
      from sklearn.preprocessing import MinMaxScaler
      from sklearn.model_selection import train_test_split
      import pickle
      from sklearn.metrics import confusion_matrix
      from sklearn import metrics
      from sklearn.model_selection import cross_validate
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.model_selection import GridSearchCV, cross_validate
      from sklearn.tree import DecisionTreeClassifier
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.model_selection import RandomizedSearchCV
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.metrics import accuracy_score, f1_score
      import numpy as np
      from sklearn.inspection import permutation_importance
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.model_selection import RandomizedSearchCV, cross_validate
      from sklearn.metrics import make_scorer, accuracy_score, f1_score
      from sklearn import linear_model
      from sklearn.metrics import r2_score
      from sklearn.linear_model import Ridge
      from sklearn.model_selection import GridSearchCV
      from sklearn.linear_model import Lasso
      import xgboost as xg
      from sklearn.metrics import mean_squared_error as MSE
      from sklearn.metrics import accuracy_score
      from sklearn.linear_model import Ridge
      # Load the serialized object from the pickle file
      with open('dur_xgb.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

      # Load the serialized object from the pickle file
      with open('label_encoder.pkl', 'rb') as file:
        label_encoders = pickle.load(file)

    html_temp = "<div class='tableauPlaceholder' id='viz1710733360364' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='IDS_560_dashboard&#47;Dashboard1' /><param name='tabs' value='yes' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ID&#47;IDS_560_dashboard&#47;Dashboard1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1710733360364');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.minWidth='1620px';vizElement.style.maxWidth='1720px';vizElement.style.width='100%';vizElement.style.minHeight='818px';vizElement.style.maxHeight='910px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.minWidth='1620px';vizElement.style.maxWidth='1720px';vizElement.style.width='100%';vizElement.style.minHeight='818px';vizElement.style.maxHeight='910px';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1250px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>"
    components.html(html_temp)

    if st.button ("Predict the Duration of the contract"):
      output = run_pred_duration(select_agencyid,select_contracttype,select_naicscode,select_pricipalplaceofperformancestate,
                select_pricipalplaceofperformancecountry,select_entity,select_performancebasedservice,select_extentcompeted,select_solicitationprocedures,
                select_localareasetaside,select_vendoraddresstatename,select_vendoraddresscountryname,select_laborstandards,
                select_vendorbusinesstypeforProfit,select_vendorbusinesstypeallawards,select_vendorbusinesstypecorprateentity,
                select_vendorbusinesstypeManufactofgoods,select_contractvalue,select_countryofprodservorigin,select_businesstype)

      st.success('The duration of the contract is {} days'.format(int(output)))
      # #st.table(output)
      # st.dataframe(output.style.highlight_max(axis=1))

    if __name__ == "__main__":
       main()

page_names_to_funcs = {
    "—": intro,
    "Predict Business Type": business_type_predict,
    "Predict Contract Value": contract_value_predict,
    "Predict Contract Duration": contract_duration_predict
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
