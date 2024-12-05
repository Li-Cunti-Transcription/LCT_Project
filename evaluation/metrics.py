import re
import difflib
import sklearn.metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


references = [
"Taddeo, le disse: si no avere chilla piscinosa,",
"che cantare, mi punia a ventre dare, e Gior-",
"getiello mazzoccare. Lo Prencepe, che s'avea",
"fatto mettere la varda da bemaguallà, manna-",
"je subeto a Zoza, se 'nce lo voleva vennere;",
"la quale respose, che n'era mercantessa, ma",
"che se voleva 'nduono, se lo pigliasse, ca ne",
"le faceva no presiento. Taddeo, che allancava",
"pe ttenere contenta la mogliere, azzò le por-",
"tasse a luce lo partoro, azzettaje l'offerta: ma",
"da llà a quatto autre juorne Zoza aperta la ca-",
"stagna, ne scette na Voccola co dudece Poleci-",
"ne d'oro, li quale puoste 'ncoppa la medesema",
"fenestra, e biste da la schiava, ne le venne go-",
"lio dall'ossa pezzelle, e cchiammato Taddeo,",
"e mostratele accossì bella cosa, le disse: si",
"chilla Voccola ne pigliare, mi punia a ventre",
"dare, e Giorgetiello mazzoccare, e Taddeo,",
"che se lassava pigliare de filatielle, e ghiocare",
"de coda da sta perra cana, mannaje de nuovo",
"a Zoza, offerennole quanto sapesse addemman-",
"nare pe priezzo de accossì bella Voccola, da",
"la quale appe la stessa resposta de primmo, che",
"'nduono se l'avesse pigliata, ca pe termene de",
"vennetta 'nce perdeva lo tiempo: e isso, che",
"non poteva farene de manco, fece dare da la",
"necessetà mazzafranca a la descrezzione; e",
"scervecchiannone sto bello voccone, restaje am-",
"misso da la liberaletà de na femmena, essenno",
"de natura tanto scarzogne, che no le vastarria-",
"no tutte le verghe, che beneno dall'India. Ma",
"passanno autre tante juorne, Zoza aprette la",
"nocella, da la quale scette fora na pipata, che",
"filava oro, cosa veramente da strasecolare, che",
"non accossì ppriesto fu posta a la medesima fe-",
"stite, chi co la cannacca, e collane; e ffattala",
"bella comme a no Sole, la mesero a na car-",
"rozza a seje cavalle, accompagnata da staffiere,",
"e da pagge de livrera; e ghionta a lo medese-",
"mo luoco dove era stata l'autra festa, agghion-",
"ze maraviglia a lo core de le ssore, e ffuoco a",
"lo pietto de lo Rrè: ma repartutase, e ghiu-",
"tole dereto lo servetore, pe non farese arrevare,",
"jettaie na vranca de perne, e de gioje, dove",
"remmase chill'ommo da bene a ppizzoliarennel-",
"le, ca non era cosa da perdere; Essa ebbe tiem-",
"po de remmorchiarese a la casa, e de spoglia-",
"rese conforme a lo ssoleto. Tornaje lo serveto-",
"re luongo luongo a lo Rrè, lo quale disse, pe'",
"l'arma de li muorte mieje, ca si tu non truo-",
"ve chesta, te faccio na 'ntosa, e te darraggio",
"tanta cauce 'nculo, quanto aje pile a sta var-",
"va. Venne l'autra festa, e sciute le ssore, essa",
"tornaje a lo dattolo, e continunanno la canzona",
"fatata, fu bestuta soperbamente, e pposta dinto",
"na carrozza d'oro co ttante serviture attuorno,",
"che pareva pottana pigliata a lo passiggio 'ntor-",
"niata de tammare; e ghiuta a fare cannavola a",
"le Ssore, se partette; e lo servetore de lo Rrè,",
"se cosette a ffilo duppio co la carrozza. Essa",
"vedendo, che sempre l'era a le ccoste, disse",
"tocca cocchiero, ed ecco se mese la carrozza a",
"ccorrere de tanta furia, e ffu cossì granne la",
"corzera, che le cascaje no chianiello, che non",
"se poteva vedere la cchiù ppentata cosa. Lo",
"servetore, che non potte jognere la carrozza,",
"che bolava, auzaje lo chianiello da terra, e lo",
"portaje a lo Rrè decennole; quanto l'era soc-",
"ceduto, lo quale pigliatolo 'nmano disse: Se"
]

pred_gpt = [
"Taddeo, le disse: si no avere chilla piscinosa,",
"che cantare, mi punia a ventre dare, e Gior-",
"gitiello mazzoccare. Lo Prencepe, che s’avea",
"fatto mettere la varda da bemaguallà, manna-",
"je subeto a Zoza, se 'nce lo voleva venere;",
"la quale respose* che n’era mercantessa, ma",
"che fe voleva n'duono, se lo pigliasse, ca ne",
"le faceva no presiento.* Taddeo, che allancava",
"pe ottenere contenta la mogliera, azzò le por-",
"tasse a luce lo partoro, azzettaje l'offerta: ma",
"da lla a quattro autre juorne Zoza aperta la ca-",
"stagna, ne scette na Voccola co dodece Polec-",
"cina d'oro, li quale proste 'ncoppa la medesema",
"finestra, e bibbe da la fchiava, ne le venne go-",
"lio dall'ossa pezzelle, e chiamannolo Taddeo",
"e mostratole accosì bella, co le disse: si",
"chilla Voccola no pigliare, mi punia a ventre",
"dare, e Giorgitiello mazzoccare, e Taddeo,",
"che se lassava pigliare de filatielle, e ghioccare",
"de coda da sta perra cana, mannaje de nuovo",
"a Zoza, offerennole quanto sapesfe addimanna-",
"nare pe prezzo de accosì bella Voccola, da la",
"quale appe la stessa resposita de primmo; che",
"n'duono se l'avesse pigliata, ca pe termine de",
"ventenatta 'nce perdeva lo tiempo: e isso, che",
"non poteva farene de manco, fece dare da la",
"necessetà mazzafranca a la defcrezzione; e",
"s'ercevechianonne sto bello voccone, reste am-",
"misso da la liberaletà de na femmena, effenno",
"de natura tanto scarzogne, che no le vasfattiria-",
"no tutte le verghe, che veneno dall’India: Ma",
"passanno autre tante juorne, Zoza aprette la",
"nocella, da la quale scette fora na pipata, che",
"filava oro, cosa veramente da sfraseloce, che",
"non accosì ppriesto fu posta a la medesema fe-",
"stite, chi co la cannacca, e collane; e sfattala",
"bella comme a no Sole, la mesero a na car-",
"rozza a fese cavalfe, accompagnata da staffiere,",
"e da pagge de l’ivera; e ghiunta a lo medése-",
"mo luoco dove era stata l’altra setta, aghioh-",
"ze maraviglia a lo core de le Sfore, e sfiuoco a",
"lo pietto de lo Rrè: ma repartutase, e ghiu-",
"tole dereto lo servetore, pe non farese arreware,",
"jettaje na vranca de perne, e de gioje, dove",
"remmasse chill’ommo da bene a ppizzoliaretenne,",
"le, ca non’era cosa da perdere; Essa ebbe tiem-",
"po de remmorchiarse a la casa, e de sffoglia-",
"rese conforme a lo sfoleto. Tornaje lo servetore",
"re luongo-luongo a lo Rrè, lo quale disse, pe",
"l’arma de li muorte mieje, ca si tu non tru-",
"ve chesta, te faccio na ’ntofa, e te darragio",
"tanta cauce ’nculo, quanto aje pile a issa var-",
"va. Venne l’altra setta, e sciute le Sfore, essa",
"tornaje a lo ottolo; e continuanno la canzona",
"sfatata, fu festuta foperbamente, e pposta dintro",
"na carrozza d’oro co ttante serviture attuorno,",
"che pareva potrana pigliata a lo passiggio ’ntor-",
"niata de tammare; e ghiuta a fare cannavola a",
"le Sfore, se partette; e lo servetore de lo Rrè,",
"se cofette a filo duppio co la carrozza. Essa",
"vedenno, che sempre l’era a le cofte, disse",
"tocca cocchiero, ed ecco se mese la carrozza a",
"correre de tanta furia, e ffu così granne la",
"cozzeria, che le scasfaje no chiainello, che non",
"se poteva vedere la cchiù appentata cosa. Lo",
"servetore, che non potte jognere la carrozza,",
"che bolava, auzaje lo chiainello da terra, e lo",
"portaje a lo Rrè decennole; quanto l’era soc-",
"ceduto, lo quale pigliatolo ’mmano disse: Se"
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
hyp_tokens = (tokenize(pred_gpt))

tokens_set = sorted(set(ref_tokens + hyp_tokens))
label_encoder = LabelEncoder()
label_encoder.fit(tokens_set)

ref_labels = label_encoder.transform(ref_tokens)
hyp_labels = label_encoder.transform(hyp_tokens)

#matrix = confusion_matrix(ref_labels, hyp_labels)
#print("confusion matrix:\n", matrix)

print(len(references))
print(len(pred_gpt))
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

# calculate diffing
def compute_diff(references_list, predicted_list, diff_file="evaluation/diff_file.txt"):
    with open(diff_file, "a", encoding="utf-8") as f:

        for i in range(len(references_list)):
            f.write(f"Processing line {i + 1}...\n")
            ref_line = references_list[i]
            pred_line = predicted_list[i]
            f.write(f"Reference: {ref_line}\n")
            f.write(f"Prediction: {pred_line}\n")
            diff = difflib.ndiff(ref_line.split(), pred_line.split())
            # casting the generator into a list
            diff_list = list(diff)
            f.write("\n".join(diff_list))
            f.write("\n" + "="*50 + "\n")

compute_diff(references, pred_gpt)





""" apostrophe = "'"
apostrophe_code = ord(apostrophe)
print(apostrophe_code)

# replace code 8127 with code 39 in hypotheses and apply confusion matrix again
hypotheses = re.sub(r"\u8127", apostrophe, pred_gpt)

print("difference set now:", exced_tokens) """

