import difflib

references = [
"LA VECCHIA SCOPERTA",
"TRATTENEMIENTO X.",
"De la Jornata I.",


"LO Rrè de Rocoaforte se nnammora de la",
"voce de na vecchia: e gabbato da no di-",
"to rezocato, la fa dormire cod'isso; ma addo-",
"natose de le rechieppe, la fa jettare pe na fe-",
"nestra, e restanno appesa a n'arvolo è ffatata",
"da sette Fate, e deventata na bellissima gio-",
"vane, lo Rrè se la piglia pe mogliere: ma l'",
"autra sore mmediosa de la fortuna soja pe ffa-",
"rese bella, se fa scortecare, e mmore.",
"No nce fu perzona, a chi n'avesse piaciuto",
"lo cunto de Ciommetella, ed appero no gusto",
"a doje sole, vedenno liberato Canneloro, e cca-",
"sticato l'Uorco, che ffaceva tanto streverio de",
"li povere cacciature; e 'ntimato l'ordene a",
"Ghiacova, che seggellasse co l'arme soje sta",
"lettera de trattenemiento, essa cossì trascorre.",


"LO 'mmarditto vizio 'ncrastato co nnuje autre",
"femmene de parere belle, nce redduce a",
"termene tale che pe nnaurare la cornice de la",
"fronte, guastammo lo quatro de la facce; pe",
"ghianchejare le ppellecchie de la carne, roi-",
"nammo ll'ossa de li diente, e ppe ddare lu-",
"ce a li miembre, coprimmo d'ombre la vi-",
"sta, che nnanze l'ora de dare tributo a lo",
"tiempo, s'apparecchiano scazzimme all'uoc-",
"chie, crespe a la facce, e defiette a le mmole;",
"ma se mmereta biasemo na giovanella, che trop-",


"po vana se dace a sse bacantarie, quanto è",
"cchiù ddegna de castigo na vecchia, che bolen-",
"no comparere co le ffigliole, se causa l'allucca",
"de la gente, e la ruina de se stessa; comme",
"so pe contareve, se mme darrite no tantillo",
"d'aurecchie.",
"S'erano raccovete dinto a no giardino, dove",
"aveva l'affacciata lo Rrè de Rocca forte, doje",
"vecchiarelle; ch'erano lo reassunto de le dde-",
"sgrazie, lo protacuollo de li sturce, lo libro",
"maggiore de la bruttezza, le cquale avevano le",
"zervole scigliate, e 'ngrifate, lo fronte 'ncrespa-",
"to, e brognoluso, le cciglia storcigliate, e rre-",
"stolose, le pparpetole chiantute, ed a ppenne-",
"ricolo, l'uocchie vizze, e scarcagnate, la facce",
"gialloteca, ed arrappata, la vocca squacquara-",
"ta, e storta, e 'nfomma la varva d'annecchia,",
"lo pietto peluso, le spalle co la contrapanzetta",
"le braccia arronchiate, le gamme sciancate, e",
"scioffate, e li piede a ccrocco: pe la quale co-",
"sa azzò no le bedesse manco lo Sole, co cchel-",
"la brutta caira, se ne stevano 'ncaforchiate din-",
"to no vascio sotto le ffenestre de chillo Signo-",
"re, lo quale era arreddutto a ttermene, che",
"non poteva fare no pideto senza dare a lo na-",
"so de ste brutte gliannole, che d'ogne poco",
"cosa 'mbroselejavano, e sse pigliavano lo tota-",
"no; mò decenno ca no giesommino cascato da",
"coppa, l'aveva 'mbrognolato lo caruso, mò ca",
"na lettera stracciata l'aveva 'ntontolato na spal-",
"la, mò ca no poco de porvere l'aveva amma-",
"tontato na coscia, tanto che ssentenno sto scas-",
"sone de dellecatezza lo Rrè, facette argomien-",
"to, che ssotto ad isso fosse la quintassenzia de",
"le ccose cenere, lo primmo taglio de le cca-",


"mumme mollise, e l'accoppatura de le ttenne-",
"rumme, pe la quale mente cosa le venne go-",
"lio da l'ossa pezzelle, e boglia da le ccata-",
"melle de l'ossa de vedere sto spanto, e cchiarire-",
"se de sto fatto, e accommenzaje a ghiettare so-",
"spire da coppa, e bascio, a rrascare senza ca-",
"tarro, e finalemente a pparlare cchiù speduto,",
"e ffora de li diente, decenno: Dove, dove te",
"nascunne, giojello, sfuorgio, isce bello de lo",
"munno? jesce jesce sole, scaglienta 'mparatore:",
"scuopre sse belle grazie, mostra sse llocernelle",
"de l'addorosa poteca d'ammore? caccia sta ca-",
"tarozzola banco accorzato de li contante de le",
"bellezze: non essere accossì scarzogna de la vi-",
"sta toja; apre le pporte a ppovero farcone,",
"famme la 'nferta si mme la vuoje fare? lassem-",
"me vedere lo stromiento da dove esce ssa bella",
"voce: fa che bea la campana, da la quale se",
"forma lo 'ntinno, famme pigliare na vista de",
"ss'auciello, non consentire, che ppecora de",
"Ponto mme pasca de nascienzo, co nnegareme",
"lo mmirare, e ccontemprare sta bellezzetudene",
"cosa; cheste, ed autre pparole deceva lo Rrè;",
"ma poteva sonare a grolia, ca le becchie ave-",
"vano 'ntompagnate l'aorecchie, la quale cosa",
"refonneva legne a lo ffuoco. E lo Rrè, che se",
"senteva comm'a ffierro scaudare a la fornace de",
"lo desederio, tenere da le ttenaglie de lo pen-",
"ziero, e mmartellare da lo maglio de lo tor-",
"miento amoruso, pe fare na chiave, che potes-",
"se aprire la cascetella de le gioje, che lo face-",
"vano morire speruto; ma non pe cchesto se",
"dette arreto, ma secotaje a mmannare suppre-",
"che, e a rrenforzare assunte, senza pigliare ma-",
"je abbiento. Tanto che le becchie, che s'era-"
]

pred_gpt = [
"LA VECCHIA SCOPERTA",
"TRATTENEMIENTO X.",
"De la Jornata I.",


"Lo Rré de Rocoaforte se 'nnammora de la",
"voce de na vecchia: e gabbato da no di-",
"ro rezocato, la fa dormire cod’ isso; ma addo-",
"natose de le rechieppe, la fa jettare pe na fe-",
"nestra, e restanno appesa a ’n arbolo è ffatata",
"da sette Fate, e deventata na bellissima gio-",
"vane, lo Rré se la piglia pe mogliere: ma l’",
"autra sore mmediosa de la fortuna soja pe ffa-",
"se bella, se fa scortecare, e mmore.",
"No nce fu perzona, a chi n’ avesse piaciuto",
"lo cunto de Ciommetella, ed appero no gusto",
"a doje sole, vedenno liberato Cannelore, e cca-",
"sticato l’ Uorco, che ffaceva tanto streverio de",
"li povere cacciature; e n’timato l’ ordene a",
"Ghiacova, che seggellasse co l’ arme soje sta",
"lettera de trattenemiento, essa cossì trascorre.",


"O 'mmarditto vizio 'neraftato co nnuje autre",
"femmene de parete belle, nce redduce a",
"termene tale che pe nnaurare la cornice de la",
"fronte, guastammo lo quatro de la facce; pe",
"ghianchejare le ppellecchie de la carne, roi-",
"nammo l’ ofsa de li diente, e ppe ddare lu-",
"ce a li miembre, coprimmo d’ ombre la vi-",
"sta, che nnanze l’ ora de dare tributo a lo",
"tiempo, s’apparecchiano scazzimme all’ uoc-",
"chie, crespe a la facce, e defiette a le mmole;",
"ma se mmereta biasemo na giovanella, che trop-",


"po vana se dace a fse bacantarie, quanto è",
"chiù ddegnia de castigo na vecchia, che bolen-",
"no comparere co le figliiole, se causa l’alluca",
"de la gente, e la ruina de se stessa; comme",
"fo pe contrareve, se mme darrite no tantillo",
"d’aurecchie.",
"S’erano raccovete dinto a no giardino, dove",
"aveva l’affacciata lo Rré de Rocca-forte, doje",
"vecchiarelle; ch’erano lo reafsunto de le dde-",
"grazìe, lo protacuollo de li sfurce, lo libro",
"maggiore de la bruttezza, le quale avevano le",
"zervole fscigliate, e ’ngrifate, lo fronte ’ncrefpa-",
"to, e brognolufo, le ciglia ftorcigliate, e rre-",
"ftolofe, le pparpetoIe chiantute, ed a ppenne-",
"ricolo, l’uocchie vizze, e fcarcagnate, la facce",
"gialloteca, ed arrappata, la vocca sfaqquara-",
"ta, e florta, e ’nfomma la varva d’annecchia,",
"lo pietto pelufo, le fpalle co la contrapanzetta,",
"le braccia arronchiate, le gamme fciancate, e",
"fcioffate, e li piede a crocco: pe la quale co-",
"fa azzò no Ie bedefse manco lo Sole, co cchel-",
"la brutta caira, fe ne ftevano ’ncafofnirate din-",
"to no vafto fotto le ffenefstre de chillo Signo-",
"re, lo quale era arredutto a ttermene, che",
"non poteva fare no pideto senza dare a lo na-",
"so de fte brutte giannole, che d’ogne poco",
"cofa ’mbrofelejavano, e fse pigliavano lo tota-",
"no; mò decenno ca no gieformino cafcato da",
"coppa, l’aveva ’mbrognolato lo carufo, mò ca",
"na Iettera ffracciata l’aveva ’ntontolato na fpal-",
"la, mò ca no poco de porvere l’aveva amma-",
"ttontato na cofcía, tanto che ffentennò fto fca-",
"fone de dellecatezza lo Rré, facette argomien-",
"to, che fotto ad isso fofse la quintafsenza de",
"Ie ccofe cenere, lo primmo taglio de Ie cca-",


"mumme mollife, e l’accoppatura de le ttenne-",
"mumme, pe la quale mente cofa le venne go-",
"lio da l’ofsa pezzelle, e boglia da le ccata-",
"melle de l’ofsa de vedere fto fpanto, e chiiarire-",
"fe de fto fatto, e accommenzaje a ghiettare fo-",
"spire da coppa, e bafcio, a rrafcare senza ca-",
"tarro, e finalmente a pparlare cchiù fpeduto,",
"e ffiora de li diente, decenno: Dove, dove te",
"nafscunne, giojello, sfuorgio, ifce bello de lo",
"mmunno? jefce jefce fole, fcaglíenta ’mparatore:",
"fscuopre fte belle grazie, moftra fte filocernelle",
"de l’addorofa poteca d’ammore? caccia fta ca-",
"tarozzola banco accorzato de li contante de le",
"bellezze: non effere accofsi fcarzogna de la vi-",
"fta toja; apre le pporte a ppovero farcone,",
"famme la ’nferta sì mme la vuoje fare? Iaffem-",
"me vedere lo fftrumento da dove efce fsa bella",
"voce: fa che bea la campana, da la quale fe",
"forma lo ’ntinno, famme pigliare na vifta de",
"fs’auciello; non consentire, che ppecora de",
"Ponto mme pafca de nafscienzo, co ’nnegareme",
"lo mmirare, e contemplare fta bellezetudene",
"cofa; chefte, ed altre pparole deceva lo Rré;",
"ma poteva fonare a grolia, ca le becchie ave-",
"vano ’ntompagnate l’aorecchie, la quale cofa",
"refonneva legne a lo ffiuoco. E lo Rré, che fe",
"fenteva comm’a fierro fscaudare a la fornace de",
"lo defiderio, tenere da le ttenaglie de lo pen-",
"ziero, e mmartellare da lo maglio de lo tor-",
"miento amorufo, pe fare na chiave, che potef-",
"fe aprire la ccfecetella de le gioje, che lo face-",
"vano morire fpeduto; ma non pe cchefto fe",
"dette arreto, ma fcootaje a mmanare fuppre-",
"che, e a rrenforzare afsunze, senza pigliare ma-",
"je abbiento. Tanto che le becchie, che s’era"
]

# calculate diffing
def compute_diff(references_list, predicted_list, diff_file="evaluation/diff_file.txt"):
    with open(diff_file, "w", encoding="utf-8") as f:

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