
def msgdecode(msgid,key):
    r = requests.session()
    r.headers["Content-Type"]="application/json; charset=utf-8"
    r.headers["Authorization"]="Bearer " + key
    response=r.get("https://api.ciscospark.com/v1/messages/"+msgid)
    response=json.loads(response.text)
    text=response["text"]
    sender=response["personId"]
    roomid=response["roomId"]
    # print "text, sender, key is ",text,sender,roomid,key
    return [text,sender,roomid,key]

def msgpost(room,text,key):  
    count=0
    p = requests.session()
    p.headers["Content-Type"]="application/json; charset=utf-8"
    p.headers["Authorization"]="Bearer "+key
    payload={"text":str(text),"markdown":str(text),"roomId":str(room)}
    print "payload is ",payload
    res=p.post("https://api.ciscospark.com/v1/messages/",json=payload)
    
    #print(res)
    #print(room,text,key)

def on_message(ws,message):
    try:
        data=json.loads(message)
        #print(data)
    except:
        data=""
        botname=""
        sender=""
    if data:

        flag=1
        botname=data["name"]
        sender=data["data"]["personEmail"]

        if(sender == "vinitbodhwani123@gmail.com"):        
            msgid=data["data"]["id"]
            decoded=decodemsg(msgid,key)
            print "decoded msge is", decoded
            # postmsg(decoded[2],decoded[0],key)

            msgid=data["data"]["id"]
            decoded=decodemsg(msgid,key)
            message=str(decoded[0]).lower()
            sender=str(decoded[1])
            roomID=str(decoded[2])
            # print("Message: "+message+ "\nSender:  " +sender+"\nRoom:   " +roomID+"\n key : " +decoded[3])  
            if sender not in talkers.keys():
                t1=fees_FSM(message,roomID)
                talkers[sender]=t1
            else:
                states=talkers[sender].possible_states[talkers[sender].state]
                for possible_answer in states:
                    # print(states)
                    if possible_answer in message:
                        flag=0
                        print("matched",possible_answer)
                        # print(talkers[sender].state)
                        talkers[sender].state=possible_answer
                        if(talkers[sender].state == 'final'):
                            try:
                                postmsg(talkers[sender].room,talkers[sender].questions[talkers[sender].state],key)
                            except:
                                talkers[sender].state="initial"
                                print "done initial state"
                                # print("state changed to fallback")
                                # print(talkers[sender])
                                postmsg(talkers[sender].room,talkers[sender].questions[talkers[sender].state],key)
                        else:                        
                            postmsg(talkers[sender].room,talkers[sender].questions[talkers[sender].state],key)
                            break
                if flag:
                    talkers[sender].state="initial"
                    # print("state changed to fallback")
                    # print(talkers[sender])
                    postmsg(talkers[sender].room,talkers[sender].questions[talkers[sender].state],key)

def on_error(ws, error):
    print error

def on_close(ws):
    print "### closed ###"

def on_open(ws):
    def run(*args):
        ws.send("subscribe:"+botname)
        while(1>0):
            time.sleep(30)
            ws.send("")
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())

    def run(*args):
        ws.send("subscribe:"+botname)
        while(1>0):
            time.sleep(30)
            ws.send("")
        ws.close()
        print("thread terminating...")
    thread.start_new_thread(run, ())

class extractapi():
	def __init__(self,filename="",load=""):
		if load!="":
			self.featuresets=self.load()
		else:
			self.filename=filename
			self.training_data=self.process_dataset()
			self.featuresets = [(self.process_sentence(n), intent) for (n, intent) in self.training_data]
			self.featuresets = [ (n, intent) for (n, intent) in self.featuresets if n]
		self.classifier= nltk.NaiveBayesClassifier.train(self.featuresets)
	def save(self):
		pickle.dump(self.featuresets, open( "featuresets", "wb" ) )
	def load(self):
		return pickle.load( open( "featuresets", "rb" ) )
	def process_dataset(self):
		df = pd.read_csv(self.filename)
		training_data = []
		for i in range(len(df)):
			training_data.append((df['Text'][i],df['Category'][i]))
		return training_data
	def bag_of_words(self,words):
		return dict([(word, True) for word in words])
	def process_sentence(self,x):
		words = nltk.tokenize.word_tokenize(x.lower()) 
		postag= nltk.pos_tag(words)
		stopwords = nltk.corpus.stopwords.words('english')
		lemmatizer = nltk.WordNetLemmatizer()
		processedwords=[]	
		for w in postag:
		    if "VB" in w[1]:
		        processedwords.append(lemmatizer.lemmatize(w[0].lower(),'v'))
		    else:
		        processedwords.append(lemmatizer.lemmatize(w[0],'n').lower())
		l=[]
		for w in processedwords:
		    if w.lower()=="not":
		        l.append(w)
		    elif w not in stopwords:
		        if (len(w)>2):
		            l.append(w)
		return self.bag_of_words(l)
	def score(self,input_sent):		
		input_sent = input_sent.lower()
		dist = self.classifier.prob_classify(self.process_sentence(input_sent))
		temp=[]
		for label in dist.samples():
		    temp.append((label, dist.prob(label)))
		return temp
	def intent(self,input_sent):
		print(input_sent)
		dist = self.classifier.classify(self.process_sentence(input_sent))
		prob = self.score(input_sent)
		prob = sorted(prob,key=lambda x:(-x[1],x[0]))
		if(prob[0][1]<0.5):
			return  "fallback"
		else:
			return dist
var1=apiextraction("dataset.csv")


if __name__ == "__main__":

    prev=""
    while(True):
        rg = requests.get("https://nlpbot.herokuapp.com/output")
        # print "rg-text is",rg.text
        # print "prev is ",prev
        if(rg.text!=prev):
            on_message("ws",rg.text)
            prev=rg.text
        if(rg.text=="STOP"):
            break    
