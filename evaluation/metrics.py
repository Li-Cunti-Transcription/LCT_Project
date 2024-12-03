import re
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


references = [
"venno dato l'utema mano a lo quatro de la",
"vita, è stato puosto drinto na sebetura, fora le",
"mmura de la Cetate, dov'è no spettaffio scrit-",
"to a na preta, che qualesevoglia femmena che",
"'nchiarrà de chianto 'ntre ghiuorne na lancella,",
"che llà mmedesemo stace appesa a no crocco,",
"lo farrà resorzetare, e pigliarrà pe mmarito:",
"e perchè è 'mpossibele, che duje uocchie omane",
"pozzano piscioliare tanto, che facciano zeppa",
"na lancella accossì granne, che leva miezo sta-",
"ro, si non fosse, commo aggio 'ntiso dicere,",
"chella Geria, che se fece a Romma Fontana",
"de lagreme: io pe vedereme delleggiata, e cof-",
"fiata da vuje, v'aggio data sta jastemma, la",
"quale prego lo Cielo, che te venga a colà pe",
"mmennetta de la 'ngiuria, che mm'è stata fat-",
"ta. Accossì decenno, sfilaje pe le grade a ba-",
"scio pe paura de quarche 'ntosa. Ma Zoza a lo",
"mmedesemo punto rommenanno, e mmazze-",
"canno le pparole de la Vecchia, le trasette a",
"la cecotena, a la catarozzola, e botato no",
"centimmolo de penziere, e no molino de dub-",
"bie sopra sto fatto, all'utemo tirata co no",
"straolo da chella passione, che cceca lo jodizio,",
"e 'ncanta lo descurzo dell'ommo, pigliatose na",
"mano de scute da li scrigne de lo Patre, se ne",
"sfilaje fora de lo Palazzo, e tanto camminaje,",
"che arrivaje a no Castiello de na Fata, co la",
"quale spaporanno lo core, essa pe ccompassione",
"de accossì bella Giovane, a la quale erano du-",
"je sperune a ffarela precepetare, e la poca aje-",
"tate, e l'ammore sopierchio a cosa scanosciuta,",
"le deze na lettera de raccommannazione a na",
"sore soja puro fatata: la quale fattole gran",
"compremiento, la matina, quanno la notte fa",
"LA GATTA CENNERENTOLA.",
"TRATTENEMIENTO VI.",
"De la Jornata I.",
"ZEzolla 'nvezzata da la Majestra ad acci-",
"dere la Matreja, e ccredenno, co ffarele",
"avere lo Patre pe mmarito, d'essere tenuta",
"cara, è pposta a la cucina; ma pe bertute de",
"le Ffate, dapò varie fortune, se guadagna no",
"Rrè pe mmarito.",
"Parzero statole l'Ascotante a ssentire lo cun-",
"to de lo Polece, e ffaccettero na dechiaratoria",
"d'asenetate a lo Rrè Catammaro, che mmese",
"a ttanto riseco lo 'nteresse de lo sango, e la",
"soccessione de lo stato pe na cosa de vrenna;",
"ed essenno tutte appilate, Antonella spilaje de",
"la manera, che ssecota.",
"SEmpre la 'nvidia ne lo maro de la male-",
"gnetate appe 'ncagno de vessiche la gualla-",
"ra, e ddove crede de vedere autro annegato a",
"mmaro, essa se trova o sott'acqua, o tozza-",
"ta a no scuoglio, comme de cierte ffigliole 'n-",
"vediose me va 'mpeziero de ve contare; sa-",
"perrite donca, che",
"Era na vota no Prencepe vidolo, lo quale",
"aveva na figliola accossì ccara, che non vedeva",
"pe d'autr'uocchie, a la quale teneva na Ma-",
"jestra prencepale, che le 'mmezzava le ccate-",
"nelle, lo punto nn' aiero, le sfilatielle, e l'afre-",
"co perciato, mostrannole tant'affezzione, che",
"non s'abbasta a ddicere. Ma essennose 'nzorato",
"XXXII.",
"Bordocchio intanto il fiume avea passato",
"Soverchiand'ogn'incontro ogni ritegno;",
"Quando del Potta, che venia, fu dato",
"Dalla Torre a Gherardo e agli altri il segno:",
"Se n'avvide Bordocchio, e rivoltato",
"Di ripassare a suoi facea disegno;",
"Ma nell'onda il destrier sotto gli cade,",
"E rimase prigion fra cento spade.",
"XXXIII.",
"Quei, ch'erano con lui dianzi passati,",
"Dal figlio di Rangon tutti fur morti;",
"E già gli altri fuggian rotti e sbandati,",
"Del mal consiglio lor, ma tardi, accorti;",
"Quando in ajuto da' vicini prati",
"Vider venir correndo i lor consorti,",
"Che del Panaro alla sinistra sponda",
"Passar più lenti, ov'è più cupa l'onda.",
"XXXIV.",
"Gian Maria della Grascia un furbacciotto,",
"Ch' era di quella squadra il Capitano,",
"Come vide fuggir dal Campo rotto",
"Quei di Bordocchio insanguinando il piano:",
"Rinfacciò lor con dispettoso motto",
"La fuga vile, e l'ardimento insano:",
"E furioso i suoi quindi spingendo",
"Fe de' nemici un potticidio orrendo."
]

hypotheses = [
"venno dato l'utema mano a lo quatro de la",
"vita, è stato puosto dritto na febetura, fora le",
"mmura de la Cetate, dov'è no spettaculo scrit-",
"to a na preta, che qualsèvoglia femmena che",
"'nchiarra de chianto 'ntre ghiuorne na lancella,",
"che lla immedesemo stace appesa a no crocco,",
"lo farà resorzetare, e pigliarra pe marito:",
"e perchè è 'mpoffibele, che duje uocchie omane",
"pozzano piscìliare tanto, che facciano zeppa",
"na lancella accossì granne, che leva miezzo sta-",
"ro, sì non fosse, commo aggio 'ntiso dicere,",
"chella Geria, che se fece a Roma Fontana",
"de lagreme: io pe vederrene deleggiate, e co-",
"sfiata da vripe, v'aggio data sta jastemma, la",
"quale prego lo Cielo, che te venga a colà pe",
"mmenetta de la 'ngiuria, che mm'è stata fat-",
"ta. Accossì decenno, sfijase pe le grade a ba-",
"sficio pe paura de quarchè 'ntofa. Ma Zoza a lo",
"mmedesemo punto rommenanno e mmrazze-",
"canno le pparole de la Vecchia, le trasferette a",
"la cecotena, a la catarozzola, e botato no",
"centrimmolo de penziere, e no molino de dub-",
"bie sopra sto fatto, all'utemo tirata co no",
"sfraolo da chella passione, che cecca lo jodizio,",
"e 'ncanta lo descurzio dell'ommo, pigliatose na",
"mano de scute da li scrigne de lo Patre, se ne",
"sfijase fora de lo Palazzo, e tanto camminaje,",
"che arrivaje a no Castiello de na Fata, co la",
"quale sfaporanno lo core, e sija pe ccompagnione",
"de accossì bella Giovane, a la quale erano du-",
"je sperune a sfirarla precipetare, e la poca aje-",
"tate, e l'ammore sfopietrico a cosa scanosciuta,",
"le deze na lettera de raccomannazione a na",
"fate soja puro sfatata: la quale fattole gran",
"comprendimento, la matina, quanno la notte fa-",
"LA GATTA CENNERENTOLA.",
"TRATTENEMIENTO VI.",
"De la Jornata I.",
"Izolla 'nvezzata da la Maiestra ad acci-",
"dere la Matreja, e credenno, co ffarse",
"avere lo Patre pe mmarito, d’essere tenuta",
"cara, è pposta a la cucina; ma pe vertute de",
"le Ffate, dapò varie fortune, se guadagna no",
"Rrè pe mmarito.",
"Parzero statole l’Ascotante a ssentire lo cun-",
"to de lo Polece, e ffaccettero na dechiaratorìa",
"d’asenetate a lo Rrè Catammarro, che nmese",
"a trunto risco lo ’nteresse de lo sango, e la",
"soccessione de lo stato pe na cosa de vrenna;",
"ed essenno tutte appilate, Antonella spilaje de",
"la manera, che ssecocta.",
"Sempre la ’nvidia ne lo maro de la male-",
"gnenate appe ’ncagno de vesfiche la gialla-",
"ra, e ddove crede de vedere autro annegato a",
"mmaro, essa se trova o sott’acqua, o tozza-",
"ta a no scuoglio, comme de cierte figliole ’n-",
"vediose me va ’mpenziero de ve contare; fa-",
"perrite donca, che",
"Era na vota no Prencepe vidolo, lo quale",
"aveva na figliole accossì ccara, che non vedeva",
"pe d’autr’uocchie, a la quale teneva na Ma-",
"jestra prencepale, che le mmezava le ccate-",
"nelle, lo punto nn’atero, le sfialitelle, e l’afra-",
"co perciato, mostranolle tant’affezzione, che",
"non s’abbassa a didicere. Ma essienno ’nzorato",
"XXXII.",
"Bordocchio intanto il fiume avea passato",
"Soverchiand’ ogn’incontro ogni ritegno;",
"Quando del Potta, che venìa, fu dato",
"Dalla Torre a Gherardo e agli altri il segno:",  
"Se n’avvide Bordocchio, e rivoltato",  
"Di ripassare a suoi facea disegno;",  
"Ma nell’onda il destrier sotto gli cadde,",  
"E rimase prigion fra cento spade.",  
"XXXIII.",
"Quei, ch’erano con lui dianzi passati,",  
"Dal figlio di Rangon tutti fur morti;",  
"E già gli altri fuggian rotti e sbandati,",  
"Del mal consiglio lor, ma tardi accorti;",  
"Quando in ajuto da’ vicini prati",  
"Vider venir correndo i lor conforti,",  
"Che del Panaro alla sinistra sponda",  
"Passàr più lenti, ov’è più cupa l’onda.",  
"XXXIV.",
"Gian Maria della Gràscia un furbacciotto,",  
"Ch’era di quella squadra il Capitano,",  
"Come vide fuggir dal Campo rotto",  
"Quei di Bordocchio infanguinando il piano;",  
"Rinfacciò lor con dispettoso motto",  
"La fuga vile, e l’ardimento insano:",  
"E furioso i suoi quindi spingendo",  
"Fe de’ nemici un potticidio orrendo."  
]

def tokenize(list_of_strings):
    tokens = []
    for string in list_of_strings:
        words = string.strip().split(" ")
        print("List of words:\n", words)
        for word in words:
            chars = list(word)
            tokens.extend(chars)
    
    return tokens

ref_tokens = (tokenize(references))
hyp_tokens = (tokenize(hypotheses))

tokens_set = sorted(set(ref_tokens + hyp_tokens))
label_encoder = LabelEncoder()
label_encoder.fit(tokens_set)

ref_labels = label_encoder.transform(ref_tokens)
hyp_labels = label_encoder.transform(hyp_tokens)

#matrix = confusion_matrix(ref_labels, hyp_labels)
#print("confusion matrix:\n", matrix)

print(len(references))
print(len(hypotheses))
print(len(ref_labels))
print(len(hyp_labels))

ref_tokens_set = set(ref_tokens)
hyp_tokens_set = set(hyp_tokens)
exced_tokens = hyp_tokens_set.difference(ref_tokens_set)
print(exced_tokens)
ref_labels_set = set(ref_labels)
hyp_labels_set = set(hyp_labels)
exced_labels = hyp_labels_set.difference(ref_labels_set)
print(exced_labels)

codes = []
for t in exced_tokens:
    code = ord(t)
    codes.append(code)


print("Codes:", codes)

apostrophe = "'"
apostrophe_code = ord(apostrophe)
print(apostrophe_code)

# replace code 8127 with code 39 in hypotheses and apply confusion matrix again
hypotheses = re.sub(r"\u8127", apostrophe, hypotheses)

print("difference set now:", exced_tokens)

