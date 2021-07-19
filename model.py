import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle


##Webscrapping from multiple URL's


        
urls = [ "https://blog.hubspot.com/sales/holiday-email-templates-salespeople",
         "https://blog.hubspot.com/sales/how-to-introduce-yourself-over-email",
         "https://blog.hubspot.com/sales/sales-email-templates-to-get-and-keep-buyers-attention",
         "https://blog.hubspot.com/sales/the-cold-email-template-that-won-16-new-b2b-customers",
        "https://blog.hubspot.com/sales/sales-email-templates-from-hubspot-reps",
        "https://blog.hubspot.com/service/customer-service-email-templates",
        "https://blog.hubspot.com/service/testimonial-request-template",
        "https://blog.hubspot.com/sales/relationship-building-email-template",
        "https://blog.hubspot.com/sales/sales-email-templates-to-use-when-prospects-arent-ready-to-buy",
        "https://blog.hubspot.com/sales/the-power-of-breakup-emails-templates-to-close-the-loop",
        "https://blog.hubspot.com/sales/sales-email-templates-guaranteed-to-get-a-response",
        "https://blog.hubspot.com/sales/sales-prospecting-email-templates-you-can-start-using-today",
        "https://blog.hubspot.com/marketing/link-building-email-templates",
        "https://blog.hubspot.com/sales/sales-follow-up-email-free-email-templates",
        "https://blog.hubspot.com/service/how-to-ask-for-referrals",
        "https://blog.hubspot.com/sales/how-to-get-a-high-quality-linkedin-recommendation-email-template",
        "https://blog.hubspot.com/sales/real-estate-email-templates",
        "https://blog.hubspot.com/sales/100k-email-templates-follow-up",
        "https://blog.hubspot.com/sales/recap-email-templates",
        "https://blog.hubspot.com/sales/how-to-craft-a-perfect-pre-meeting-email-template",
        "https://blog.hubspot.com/marketing/email-templates-agency-communication",
        "https://blog.hubspot.com/sales/spring-themed-sales-emails",
        "https://blog.hubspot.com/sales/sales-email-templates-create-urgency",
        "https://blog.hubspot.com/service/how-to-ask-for-referrals",
        "https://blog.hubspot.com/sales/no-show-prospects-templates",
        "https://blog.hubspot.com/sales/new-year-sales-email-templates",
        "https://blog.hubspot.com/sales/follow-up-sales-email-templates-instead-checking-in",
        "https://blog.hubspot.com/sales/drip-emails-opens",
        "https://blog.hubspot.com/sales/email-templates-virtual-assistant?scriptPath=bundles%2Fapp.js&inpageEditorUI=true&preview_key=pSOAWoAu&cssPath=bundles%2Fapp.css&hubs_signup-url=preview.hs-sites.com%2F_hcms%2Fpreview%2Fcontent%2F5241498400&hubs_signup-cta=null&cacheBust=1594720451930&_preview=true&portalId=53&benderPackage=InpageEditorUI&staticVersion=static-1.22392&pix=sy_0_0",
        "https://blog.hubspot.com/sales/fall-sales-email-templates",
        "https://blog.hubspot.com/service/welcome-email-template",
        "https://blog.hubspot.com/sales/sales-email-templates-that-prove-flattery-will-get-you-everywhere",
        "https://blog.hubspot.com/service/apology-letter-to-customers",
        "https://blog.hubspot.com/sales/valentines-day-email-templates-sales",
        "https://blog.hubspot.com/sales/ask-for-email-introduction",
        "https://blog.hubspot.com/sales/the-phrase-that-poisons-sales-follow-up-emails",
        "https://blog.hubspot.com/sales/thank-you-for-your-consideration",
        "https://blog.hubspot.com/sales/thank-you-in-advance-alternatives",
        "https://blog.hubspot.com/sales/unconventional-sales-email-templates"

        ]
template = []
allTemplates = {}
for index,url in enumerate(urls):
    html = requests.get(url).text
    time.sleep(2)
    soup = BeautifulSoup(html, "html.parser")
    tag = ''
    if 'holiday' in url :
       tag = 'Formal_Holiday_Vacation_Leaves'
    elif 'yourself-over-email' in url:
       tag = 'Formal_Self_introduction_at_new_job '
    elif 'buyers-attention' in url:
       tag = 'Business_to_attract_buyer_attention'
    elif 'b2b' in url:
       tag = 'Business_Cold_email_to_B2B_customer'
    elif 'hubspot-reps' in url:
       tag = 'Service_Sales'
    elif 'customer-service' in url:
       tag = 'Business_Customer_service'
    # elif 'meeting-networking' in url:
    #    tag = 'Business_Follow_up_after_meeting'
    elif 'testimonial-request' in url:
       tag = 'Service_Testimonial_request '
    elif 'relationship-building' in url:
       tag = 'Marketing_Sales_Relationship_building'
    elif 'ready-to-buy' in url:
       tag = 'Sales_when_prospects_not_ready_to_buy'
    elif 'close-the-loop' in url:
       tag = 'Business_The_power_of_breakup_email_to_close_the_loop'
    elif 'get-a-response' in url:
       tag = 'Marketing_guaranteed_to_get_a_response'
    elif 'prospecting-email' in url:
       tag = 'Service_Sales_prospecting'
    elif 'link-building-email' in url:
       tag = 'Marketing_Link_building_email'
    elif 'email-free' in url:
       tag = 'Service_Sales_Follow_up'
    elif 'for-referrals' in url:
       tag = 'Service_Referrals'
    elif 'linkedin-recommendation' in url:
       tag = 'Formal_Recommendation'
    elif 'estate-email' in url:
       tag = 'Business_Real_Estate'
    elif '100k-email' in url:
       tag = 'Sales_Follow_up'
    elif 'recap-email' in url:
       tag = 'Sales_Recap'
    elif 'pre-meeting-email' in url:
       tag = 'Formal_Craft_a_perfect_Pre-meeting'
    # elif 'email-after-interview' in url:
    #    tag = 'Formal_Follow_up_after_interview'
    elif 'agency-communication' in url:
       tag = 'Marketing_Agency_communication'
    elif 'spring-themed-sales' in url:
       tag = 'Sales_Spring_themed_emails'
    elif 'create-urgency' in url:
       tag = 'Sales_Create_urgency'
    elif 'ask-for-referrals' in url:
       tag = 'Service_Asking_for_referrals'
    elif 'no-show-prospects' in url:
       tag = 'Sales_No_show_prospect'
    elif 'new-year-sales' in url:
       tag = 'Formal_New_Year'
    elif 'instead-checking-in' in url:
       tag = 'Marketing_Follow_up'
    elif 'drip-emails-opens' in url:
       tag = 'Formal_Drip_emails_opens'
    elif 'virtual-assistant' in url:
       tag = 'Formal_virtual_assistant'
    elif 'fall-sales' in url:
       tag = 'Sales_fall_sales'
    elif 'welcome-email' in url:
       tag = 'Service_Welcome'
    elif 'prove-flattery' in url:
       tag = 'Sales_email_that_prove_flattery_will_get_everywhere'
    elif 'apology-letter-to-customers' in url:
       tag = 'Service_Apology_to_customer'
    elif 'valentines-day' in url:
       tag = "Formal_Valentine's_day"
    elif 'email-introduction' in url:
       tag = 'Formal_Asking_for_an_introduction'
    elif 'poisons-sales' in url:
       tag = 'Formal_Sorry_to_bother'
    elif 'consideration' in url:
       tag = 'Service_Thank_you_for_consideration'
    elif 'thank-you-in-advance' in url:
       tag = 'Thank_you__in_advance'
    elif 'unconventional-sales' in url:
       tag = 'Sales_Unconventional'

    print( "tag selected ",tag )

    if tag == 'Sales_when_prospects_not_ready_to_buy' or tag == 'Marketing_Link_building_email' or tag == 'Marketing_Agency_communication' or tag == 'Service_Thank_you_for_consideration' or tag == 'Sales_email_that_prove_flattery_will_get_everywhere':
      print('inside  if')
      allTemplates[tag] = soup.find_all('div', attrs={'style':"font-size: 14px; line-height: 1.5em; border: 1px solid #dddddd; margin-right: 10px; margin-top: 0px; padding: 10px 10px 10px 10px; border-top-left-radius: 1px; border-top-right-radius: 1px; border-bottom-right-radius: 1px; border-bottom-left-radius: 1px; text-align: left; background-color: #ffffff;"})
   
    else:
      print('inside else')
      allTemplates[tag] = soup.find_all('div', attrs={'class':["wt-blog__email-ui__body"]})
      



##Print all templates

for k, v in allTemplates.items():
  #print(k,':',v)
    for j in allTemplates.get(k):
        print(k,':',j.text)




### CSV creation
import csv

with open('allEmailTempates.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    i = 1
    writer.writerow(["id","tags","templates"])
    for k, v in allTemplates.items():
      for j in allTemplates.get(k):
          writer.writerow([i,k,j.text])
          i = i+1
        

email_template = pd.read_csv('allEmailTempates.csv',usecols= ["tags","templates"]) 


email_template.head(n = 100)



intents_file = open('intents.json',encoding='utf-8').read()
intents = json.loads(intents_file)

# this code takes all words in patterns along with puntuncation and stop words and add to the words list 
# and also adds each pattern along with respective tag to 'documents'
# and all types of tag into one list called 'classes'

words=[]
len(words)
classes = []
documents = []
ignore_letters = ['t','u','r','k','f','a','+1','-1','!', '?', ',', '.','(',')','[',']','&',"'s",':','%',"'",'/','-','2','4','a/b','x','z','‘','’']


for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        #add documents in the corpus
        documents.append((pattern, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
#collecting root words(lemmatization) from 'words' list and saving after sorting 
# sorted 'classes' (nothing but tags in data) and saving


# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
#saving words and classes as pickle format to load later
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words=[]
    word = nltk.word_tokenize(doc[0])
    pattern_words.extend(word)
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
     
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
train_x=np.array(train_x)
train_y=np.array(train_y)
# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal tonumber of intents to predict output intent with softmax
import tensorflow as tf
model = Sequential()
model.add(tf.keras.layers.Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
model.save('email_template.h5', hist)