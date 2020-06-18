import json
import pandas as pd
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions


url_api = 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/8358a9cb-b7e2-46a1-832f-7452ff63d082'
api_key = '2xDdcc0s8RvOJmQjfgkMZuepL-wjt-4_51w0eAaXBcXi'

analise_text = '''Desde o início, a diretoria do Nacional não se mostrou disposta a fazer negócio com o Palmeiras. O principal argumento dos colombianos era a pendência que ainda existe da contratação do atacante Borja, em 2017, fato que é contestado pelo clube brasileiro.
Em fevereiro, o presidente Juan David Pérez deu uma entrevista declarando que não venderia Muñoz ao Palmeiras por conta desse motivo. Ele chegou a dizer que "aqui não se castra um cachorro duas vezes", em referência ao imbróglio envolvendo a negociação de Borja.
A transferência para o Genk foi confirmada pelos dois clubes envolvidos. Muñoz ainda fará exames médicos, quando houver permissão por conta da pandemia do coronavírus, antes de fechar o contrato. Ele era o capitão do Atlético Nacional.
Com isso, o Palmeiras segue com Marcos Rocha e Mayke como opções para a lateral direita no elenco.'''


#Todo processo de autenticação e request é feito usando a biblioteca da IBM. 
#Para mais informação deve-se acessar a documentação da NLU Watson.
#link => https://cloud.ibm.com/apidocs/natural-language-understanding?code=python#sentiment

auth = IAMAuthenticator(f'{api_key}')
nlu = NaturalLanguageUnderstandingV1(version='2019-07-12', authenticator=auth)
nlu.set_service_url(f'{url_api}')
response = nlu.analyze(text=f'{analise_text}', features=Features(entities=EntitiesOptions(emotion=True, sentiment=True, limit=5), keywords=KeywordsOptions(emotion=True, sentiment=True,limit=10))).get_result()


#A chamada para a <API> esta me retornando um <dict> ai inves de um <JSON>. 
#Por isso primeiro converto com <DUMPS> o <dict> para <json>.
#Depois converto o <JSON> para <Python Object> com <loads>. 

js = json.dumps(response, indent=2)
extract = json.loads(js)  

# List Comprehension
text = [t['text'] for t in extract['keywords']] 
sent = [s['sentiment'] for s in extract['keywords']]

# DataFrame
df = pd.DataFrame(index=['text', 'sent'], data=[text, sent])
print(df)



