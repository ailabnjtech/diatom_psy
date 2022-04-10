import os
from os import getcwd

#---------------------------------------------#
#   训练前一定要注意修改classes
#   种类顺序需要和model_data下的txt一样
#---------------------------------------------#
classes =['Acanthoceras zachariasii','Achnanthes brevipes','Achnanthes coarctata','Achnanthes felinophila','Achnanthes inflata','Achnanthes longboardia','Achnanthes mauiensis','Achnanthes rostellata','Achnanthes tumescens','Achnanthes undulorostrata','Achnanthidium affine','Achnanthidium alpestre','Achnanthidium atomus','Achnanthidium catenattum','Achnanthidium crassum','Achnanthidium deflexum','Achnanthidium delmontii','Achnanthidium druartii','Achnanthidium duthiei','Achnanthidium eutrophilum','Achnanthidium exiguum','Achnanthidium gracillimum','Achnanthidium kranzii','Achnanthidium latecephalum','Achnanthidium minutissimum','Achnanthidium pyrenaicum','Achnanthidium reimeri','Achnanthidium rivulare','Achnanthidium rosenstockii','Achnanthidium subhudsonis','Achnanthidium subhudsonis var. kraeuselii','Actinella punctata','Actinocyclus normanii','Adlafia detenta','Adlafia multnomahii','Adlafia suchlandtii','Amphipleura pellucida','Amphora bicapitata','Amphora calumetica','Amphora copulata','Amphora delphinea var. minor','Amphora minutissima','Amphora montana','Amphora oligotraphenta','Amphora ovalis','Amphora pediculus','Aneumastus carolinianus','Aneumastus minor','Aneumastus pseudotusculus','Aneumastus rostratus','Aneumastus tusculus','Anomoeoneis capitata','Anomoeoneis costata','Anomoeoneis fogedii','Anomoeoneis monoensis','Anomoeoneis sculpta','Anomoeoneis sphaerophora','Anomoeoneis sphaerophora  f. rostrata','Asterionella A. ralfsii var. americana','Asterionella formosa','Aulacoseira alpigena','Aulacoseira ambigua','Aulacoseira canadensis','Aulacoseira crassipunctata','Aulacoseira granulata','Aulacoseira granulata var. angustissima','Aulacoseira herzogii','Aulacoseira humilis','Aulacoseira islandica','Aulacoseira italica','Aulacoseira lirata','Aulacoseira muzzanensis','Aulacoseira nivalis','Aulacoseira nivaloides','Aulacoseira nygaardii','Aulacoseira pusilla','Aulacoseira subarctica','Aulacoseira tenella','Aulacoseira valida','Bacilaria paxillifer','Biremis circumtexta','Biremis undulata','Boreozonacola hustedtii','Boreozonacola olympica','Brachysira arctoborealis','Brachysira brebissonii','Brachysira follis','Brachysira hannae','Brachysira microcephala','Brachysira neoacuta','Brachysira ocalanensis','Brachysira serians','Brachysira styriaca','Brachysira subirawanae','Brachysira subtilis','Brachysira vitrea','Brachysira zellensis','Brebissonia lanceolata','Caloneis amphisbaena','Caloneis bacillum','Caloneis fusus','Caloneis lewisii','Caloneis silicula','Campylodiscus hibernicus','Capartogramma crucicula','Cavinula cocconeiformis','Cavinula davisiae','Cavinula jaernefeltii','Cavinula lapidosa','Cavinula maculata','Cavinula pseudoscutiformis','Cavinula pusio','Cavinula scutelloides','Cavinula scutiformis','Cavinula variostriata','Cavinula vincentii','Chaetoceros lauderi','Chaetoceros muelleri','Chamaepinnularia hassiaca','Chamaepinnularia mediocris','Chamaepinnularia soehrensis','Chamaepinnularia witkowskii','Cocconeis cascadensis','Cocconeis fluviatilis','Cocconeis grovei','Cocconeis klamathensis','Cocconeis neodiminuta','Cocconeis pediculus','Cocconeis placentula','Cocconeis placentula sensu lato','Cocconeis pseudothumensis','Cocconeis rugosa','Cosmioneis citriformis','Cosmioneis hawaiiensis','Cosmioneis reimeri','Craticula accomoda','Craticula accomodiformis','Craticula acidoclinata','Craticula ambigua','Craticula buderi','Craticula citrus','Craticula coloradensis','Craticula cuspidata','Craticula halophila','Craticula johnstoniae','Craticula molestiformis','Craticula pampeana','Craticula riparia','Craticula sardiniana','Craticula subminuscula','Ctenophora pulchella','Cyclostephanos dubius','Cyclostephanos invisitatus','Cyclostephanos tholiformis','Cyclotella atomus','Cyclotella distinguenda','Cyclotella gamma','Cyclotella meneghiniana','Cyclotella quillensis','Cylindrotheca fusiformis','Cymatopleura internationale','Cymbella affiniformis','Cymbella affinis','Cymbella alpestris','Cymbella amplificata','Cymbella aspera','Cymbella blinnii','Cymbella compacta','Cymbella cosleyi','Cymbella cymbiformis','Cymbella designata','Cymbella excisa','Cymbella excisiformis','Cymbella fontinalis','Cymbella hantzschiana','Cymbella hustedtii','Cymbella janischii','Cymbella laevis','Cymbella lanceolata','Cymbella maggiana','Cymbella mexicana','Cymbella mexicana var. kamtschatica','Cymbella neocistula','Cymbella neocistula var. islandica','Cymbella neoleptoceros','Cymbella parva','Cymbella perfossilis','Cymbella proxima','Cymbella proxima  f. gravida','Cymbella rumrichae','Cymbella stigmaphora','Cymbella subleptoceros','Cymbella subturgidula','Cymbella tropica','Cymbella tumida','Cymbella turgidula','Cymbella vulgata','Cymbellonitzschia diluviana','Cymbopleura amphicephala','Cymbopleura anglica','Cymbopleura angustata','Cymbopleura apiculata','Cymbopleura austriaca','Cymbopleura crassipunctata','Cymbopleura edlundii','Cymbopleura elliptica','Cymbopleura florentina','Cymbopleura fluminea','Cymbopleura frequens','Cymbopleura heinii','Cymbopleura hybrida','Cymbopleura inaequalis','Cymbopleura incerta','Cymbopleura incertiformis','Cymbopleura incertiformis var. laterostrata','Cymbopleura incertiformis var. linearis','Cymbopleura lapponica','Cymbopleura laszlorum','Cymbopleura lata','Cymbopleura maggieae','Cymbopleura metzeltinii','Cymbopleura naviculiformis','Cymbopleura oblongata','Cymbopleura perprocera','Cymbopleura rainierensis','Cymbopleura rupicola','Cymbopleura stauroneiformis','Cymbopleura subaequalis','Cymbopleura subcuspidata','Cymbopleura sublanceolata','Cymbopleura subrostrata','Cymbopleura tundraphila','Cymbopleura tynnii','Decussata placenta','Delicata alpestris','Delicata canadensis','Delicata delicatula','Delicata montana','Denticula creticola','Denticula subtilis','Denticula tenuis','Diadesmis confervacea','Diatoma ehrenbergii','Diatoma moniliformis','Diatoma tenuis','Diatoma vulgaris','Diatomella balfouriana','Didymosphenia geminata','Diploneis abscondita','Diploneis boldtiana','Diploneis calcicolafrequens','Diploneis calcilacustris','Diploneis elliptica','Diploneis finnica','Diploneis krammeri','Diploneis mollenhaueri','Diploneis ovalis','Diploneis parma','Diploneis potapovae','Diploneis puella','Diploneis puellafallax','Diploneis submarginestriata','Diploneis texana','Diprora haenaensis','Discostella asterocostata','Discostella lakuskarluki','Discostella pseudostelligera','Discostella stelligera','Distrionella incognita','Ellerbeckia arenaria','Encyonema appalachianum','Encyonema auerswaldii','Encyonema evergladianum','Encyonema gracile','Encyonema hamsherae','Encyonema hebridicum','Encyonema lange-bertalotii','Encyonema latum','Encyonema leibleinii','Encyonema lunatum','Encyonema minutum','Encyonema minutum var. pseudogracilis','Encyonema montana','Encyonema neogracile','Encyonema nicafei','Encyonema norvegicum','Encyonema obscurum','Encyonema paucistriatum','Encyonema pergracile','Encyonema procerum','Encyonema reichardtii','Encyonema reimeri','Encyonema sibericum','Encyonema silesiacum','Encyonema temperei','Encyonema triangulum','Encyonema yellowstonianum','Encyonopsis aequaliformis','Encyonopsis albertana','Encyonopsis alpina','Encyonopsis anacondae','Encyonopsis bobmarshallensis','Encyonopsis cesatiformis','Encyonopsis cesatii','Encyonopsis czarneckii','Encyonopsis dakotae','Encyonopsis descripta','Encyonopsis descriptiformis','Encyonopsis falaisensis','Encyonopsis hustedtii','Encyonopsis krammeri','Encyonopsis kutenaiorum','Encyonopsis lacusalpini','Encyonopsis lacuscaerulei','Encyonopsis medicinalis','Encyonopsis microcephala','Encyonopsis minuta','Encyonopsis montana','Encyonopsis neerlandica','Encyonopsis perborealis','Encyonopsis robusta','Encyonopsis stafsholtii','Encyonopsis subminuta','Encyonopsis thumensis','Entomoneis alata','Entomoneis ornata','Entomoneis paludosa','Entomoneis punctulata','Envekadea metzeltinii','Envekadea pachycephala','Envekadea vanlandinghamii','Eolimna aboensis','Eolimna metafarta','Epithemia adnata','Epithemia alpestris','Epithemia argus','Epithemia gibba','Epithemia gibberula','Epithemia musculus','Epithemia reichelti','Epithemia smithii','Epithemia sorex','Epithemia turgida','Eucocconeis alpestris','Eucocconeis americana','Eucocconeis depressa','Eucocconeis flexella','Eucocconeis laevis','Eunotia areniverma','Eunotia bactriana','Eunotia bidens','Eunotia bidentula','Eunotia bilii','Eunotia bilunaris','Eunotia boomsma','Eunotia canicula','Eunotia cataractarum','Eunotia charliereimeri','Eunotia diadema','Eunotia enischna','Eunotia eruca','Eunotia exigua','Eunotia faba','Eunotia formica','Eunotia gibbosa','Eunotia hexaglyphis','Eunotia horstii','Eunotia incisa','Eunotia kociolekii','Eunotia lewisii','Eunotia macroglossa','Eunotia metamonodon','Eunotia microcephala','Eunotia minor','Eunotia minutula','Eunotia montuosa','Eunotia mucophila','Eunotia mydohaimasiae','Eunotia naegelii','Eunotia novaeangliae','Eunotia nymanniana','Eunotia obliquestriata','Eunotia orthohedra','Eunotia paludosa','Eunotia panda','Eunotia papilioforma','Eunotia pectinalis','Eunotia perpusilla','Eunotia rhomboidea','Eunotia richbuttensis','Eunotia rushforthii','Eunotia sarraceniae','Eunotia serra','Eunotia spatulata','Eunotia subherkiniensis','Eunotia sudetica','Eunotia superbidens','Eunotia tauntoniensis','Eunotia tenella','Eunotia tetraodon','Eunotia trinacria','Eunotia triodon','Eunotia zasuminensis','Eupodiscus radiatus','Fallacia californica','Fallacia hummii','Fallacia latelongitudinalis','Fallacia pygmaea','Fallacia subhamulata','Fallacia tenera','Fallacia vitrea','Fragilaria amphicephaloides','Fragilaria crotonensis','Fragilaria mesolepta','Fragilaria pennsylvanica','Fragilaria recapitellata','Fragilaria socia','Fragilaria synegrotesca','Fragilaria tenera','Fragilaria vaucheriae','Fragilariforma acidobiontica','Fragilariforma bicapitata','Fragilariforma constricta','Fragilariforma horstii','Fragilariforma marylandica','Fragilariforma nitzschioides','Fragilariforma polygonata','Fragilariforma virescens','Frickea lewisiana','Frustulia amosseana','Frustulia amphipleuroides','Frustulia asiatica','Frustulia bahlsii','Frustulia capitata','Frustulia crassinervia','Frustulia creuzburgensis','Frustulia esandalliae','Frustulia inculta','Frustulia krammeri','Frustulia latita','Frustulia neomundana','Frustulia pseudomagaliesmontana','Frustulia quadrisinuata','Frustulia rexii','Frustulia saxonica','Frustulia soror','Frustulia vulgaris','Geissleria acceptata','Geissleria cascadensis','Geissleria decussis','Geissleria ignota','Geissleria kriegeri','Geissleria lateropunctata','Geissleria punctifera','Genkalia digitulus','Gliwiczia calcar','Gomphoneis eriense','Gomphoneis eriense var. angularis','Gomphoneis eriense var. apiculata','Gomphoneis eriense var. variabilis','Gomphoneis herculeana','Gomphoneis herculeana var. abundans','Gomphoneis herculeana var. loweii','Gomphoneis mammilla','Gomphoneis minuta','Gomphoneis oreophila','Gomphoneis pseudo-okunoi','Gomphoneis septa','Gomphoneis trullata','Gomphonella olivacea','Gomphonema acuminatum','Gomphonema amerhombicum','Gomphonema americobtusatum','Gomphonema apicatum','Gomphonema apuncto','Gomphonema brebissonii','Gomphonema californicum','Gomphonema caperatum','Gomphonema christenseni','Gomphonema consector','Gomphonema coronatum','Gomphonema distans','Gomphonema duplipunctatum','Gomphonema eileencoxiae','Gomphonema elongatum','Gomphonema freesei','Gomphonema germainii','Gomphonema gibba','Gomphonema incognitum','Gomphonema insularum','Gomphonema johnsonii','Gomphonema kobayasii','Gomphonema lagenula','Gomphonema louisiananum','Gomphonema manubrium','Gomphonema mehleri','Gomphonema mexicanum','Gomphonema montezumense','Gomphonema nathorstii','Gomphonema olivaceoides var. densestriata','Gomphonema olivaceoides var. hutchinsoniana','Gomphonema parvulum','Gomphonema pseudosphaerophorum','Gomphonema pusillum','Gomphonema pygmaeum','Gomphonema reimeri','Gomphonema sarcophagus','Gomphonema semiapertum','Gomphonema sierranum','Gomphonema sphaerophorum','Gomphonema submehleri','Gomphonema superiorense','Gomphonema truncatum','Gomphonema turgidum','Gomphonema turris','Gomphonema variostriatum','Gomphonema ventricosum','Gomphosinica geitleri','Gomphosphenia grovei','Gomphosphenia lingulatiformis','Gomphosphenia stoermeri','Grunowia sinuata','Grunowia solgensis','Grunowia tabellaria','Gyrosigma acuminatum','Gyrosigma attenuatum','Gyrosigma obscurum','Halamphora coffeaeformis','Halamphora elongata','Halamphora latecostata','Halamphora montana','Halamphora normanii','Halamphora oligotraphenta','Halamphora submontana','Halamphora subtilis','Halamphora thumensis','Halamphora veneta','Hannaea arcus','Hannaea superiorensis','Hippodonta capitata','Hippodonta capitata subsp. iberoamericana','Hippodonta coxiae','Hippodonta gravistriata','Hippodonta hungarica','Hippodonta pseudacceptata','Humidophila contenta','Humidophila pantropica','Humidophila perpusilla','Humidophila schmassmanni','Humidophila undulata','Hydrosera whampoensis','Hygropetra balfouriana','Iconella hibernica','Karayevia amoena','Karayevia clevei','Karayevia laterostrata','Karayevia nitidiformis','Karayevia ploenensis var. gessneri','Karayevia suchlandtii','Kobayasiella jaagii','Kobayasiella micropunctata','Kobayasiella okadae','Kobayasiella parasubtilissima','Kobayasiella subtilissima','Krasskella kriegeriana','Kurtkrammeria aequalis','Kurtkrammeria lacusglacialis','Kurtkrammeria neoamphioxys','Kurtkrammeria stodderi','Kurtkrammeria subspicula','Kurtkrammeria treinishii','Kurtkrammeria weilandii','Lacustriella lacustris','Lemnicola hungarica','Lindavia affinis','Lindavia antiqua','Lindavia bodanica','Lindavia comensis','Lindavia delicatula','Lindavia eriensis','Lindavia intermedia','Lindavia michiganiana','Lindavia ocellata','Lindavia praetermissa','Lindavia radiosa','Lindavia rossii','Luticola arctica','Luticola goeppertiana','Luticola minor','Luticola mobiliensis','Luticola simplex','Luticola stigma','Luticola ventricosa','Mastogloia albertii','Mastogloia calcarea','Mastogloia elliptica','Mastogloia grevillei','Mastogloia lacustris','Mastogloia pseudosmithii','Mastogloia pumila','Mayamaea cahabaensis','Melosira dickiei','Melosira normannii','Melosira undulata','Melosira varians','Meridion alansmithii','Meridion anceps','Meridion circulare','Meridion circulare var. constrictum','Meridion lineare','Microcostatus krasskei','Muelleria agnellus','Muelleria gibbula','Muelleria spauldingiae','Muelleria tetonensis','Navicula aitchelbee','Navicula angusta','Navicula antonii','Navicula aurora','Navicula canalis','Navicula capitatoradiata','Navicula cari','Navicula caroliniae','Navicula caterva','Navicula cincta','Navicula cryptocephala','Navicula cryptocephaloides','Navicula cryptofallax','Navicula duerrenbergiana','Navicula eileeniae','Navicula elsoniana','Navicula erifuga','Navicula escambia','Navicula flatheadensis','Navicula freesei','Navicula galloae','Navicula genovefae','Navicula germainii','Navicula geronimensis','Navicula goersii','Navicula gregaria','Navicula harmoniae','Navicula hasta','Navicula hodgeana','Navicula ingenua','Navicula kotschyi','Navicula lanceolata','Navicula leptostriata','Navicula libonensis','Navicula longicephala','Navicula ludloviana','Navicula lundii','Navicula margalithii','Navicula metareichardtiana','Navicula notha','Navicula nunivakiana','Navicula oblonga','Navicula oppugnata','Navicula pelliculosa','Navicula peregrina','Navicula peregrinopsis','Navicula perotii','Navicula piercei','Navicula pseudolanceolata','Navicula radiosa','Navicula recens','Navicula reinhardtii','Navicula rhynchocephala','Navicula rhynchotella','Navicula rostellata','Navicula salinarum','Navicula schweigeri','Navicula seibigiana','Navicula slesvicensis','Navicula sovereignii','Navicula staffordiae','Navicula streckerae','Navicula subconcentrica','Navicula subrhynchocephala','Navicula subrostellata','Navicula subwalkeri','Navicula supleeorum','Navicula symmetrica','Navicula trilatera','Navicula tripunctata','Navicula trivialis','Navicula vaneei','Navicula venerablis','Navicula veneta','Navicula vilaplanii','Navicula viridula','Navicula viridulacalcis subsp. neomundana','Navicula volcanica','Navicula vulpina','Navicula walkeri','Navicula weberi','Navicula whitefishensis','Navicula wildii','Navicula winona','Navicymbula pusilla','Neidiomorpha binodiformis','Neidiopsis hamiltonii','Neidiopsis levanderi','Neidiopsis weilandii','Neidiopsis wulffii','Neidium bobmarshallensis','Neidium densestriatum','Neidium distinctepunctatum','Neidium fogedii','Neidium hitchcockii','Neidium pseudodensestriatum','Neidium sacoense','Neidium undulatum','Ninastrelnikovia gibbosa','Nitzschia acicularis','Nitzschia acidoclinata','Nitzschia alpina','Nitzschia amphibia','Nitzschia angustata','Nitzschia balcanica','Nitzschia biacrula','Nitzschia brevissima','Nitzschia clausii','Nitzschia columbiana','Nitzschia communis','Nitzschia desertorum','Nitzschia dissipata','Nitzschia exilis','Nitzschia filiformis','Nitzschia fonticola','Nitzschia fonticoloides','Nitzschia incognita','Nitzschia innominata','Nitzschia kurzeana','Nitzschia liebethruthii','Nitzschia linearis','Nitzschia microcephala','Nitzschia minuta','Nitzschia oligotraphenta','Nitzschia oregona','Nitzschia palea','Nitzschia palea var. debilis','Nitzschia palea var. tenuirostris','Nitzschia paleacea','Nitzschia perminuta','Nitzschia recta','Nitzschia regula var. robusta','Nitzschia reversa','Nitzschia semirobusta','Nitzschia serpentiraphe','Nitzschia sigma','Nitzschia sigmoidea','Nitzschia siliqua','Nitzschia sociabilis','Nitzschia soratensis','Nitzschia valdecostata','Nupela carolina','Nupela decipiens','Nupela elegantula','Nupela fennica','Nupela frezelii','Nupela impexiformis','Nupela lapidosa','Nupela neglecta','Nupela pennsylvanica','Nupela poconoensis','Nupela potapovae','Nupela scissura','Nupela subrostrata','Nupela tenuicephala','Nupela vitiosa','Nupela wellneri','Odontidium hyemale','Odontidium mesodon','Orthoseira roeseana','Oxyneis binalis','Oxyneis binalis var. elliptica','Peronia fibula','Phaeodactylum tricornutum','Pinnularia acrosphaeria','Pinnularia borealis','Pinnularia borealis var. lanceolata','Pinnularia brauniana','Pinnularia brebissonii','Pinnularia cardinaliculus','Pinnularia cuneicephala','Pinnularia decrescens','Pinnularia divergentissima','Pinnularia formica','Pinnularia gigas','Pinnularia krammeri','Pinnularia lata','Pinnularia marchica','Pinnularia microstauron','Pinnularia nodosa','Pinnularia nodosa var. percapitata','Pinnularia parvulissima','Pinnularia polyonca var. stidolphii','Pinnularia rabenhorstii','Pinnularia rhombarea','Pinnularia rumrichae','Pinnularia saprophila','Pinnularia scotica','Pinnularia streptoraphe','Pinnularia suchlandtii','Pinnularia turfosiphila','Pinnularia turgidula','Pinnularia undula','Pinnularia undula var. major','Placoneis abiskoensis','Placoneis amphibola','Placoneis anglophila','Placoneis elginensis','Placoneis explanata','Placoneis gastrum','Plagiotropis arizonica','Plagiotropis lepidoptera var. proboscidea','Planothidium abbreviatum','Planothidium amphibium','Planothidium apiculatum','Planothidium biporomum','Planothidium delicatulum','Planothidium frequentissimum','Planothidium holstii','Planothidium incuriatum','Planothidium joursacense','Planothidium lanceolatoides','Planothidium lanceolatum','Planothidium potapovae','Planothidium reichardtii','Planothidium rostratoholarcticum','Planothidium sheathii','Platessa bahlsii','Platessa conspicua','Platessa hustedtii','Platessa kingstonii','Platessa lutheri','Platessa oblongella','Platessa stewartii','Platessa strelnikovae','Playaensis circumfimbria','Pleurosigma inflatum','Pleurosira laevis','Prestauroneis integra','Prestauroneis protracta','Proschkinia browderiana','Psammothidium acidoclinatum','Psammothidium alpinum','Psammothidium bioretii','Psammothidium chlidanos','Psammothidium daonense','Psammothidium didymum','Psammothidium harveyi','Psammothidium helveticum','Psammothidium lacustre','Psammothidium lauenburgianum','Psammothidium levanderi','Psammothidium microscopicum','Psammothidium nivale','Psammothidium obliquum','Psammothidium pennsylvanicum','Psammothidium rossii','Psammothidium scoticum','Psammothidium semiapertum','Psammothidium subatomoides','Pseudofallacia monoculata','Pseudostaurosira brevistriata','Punctastriata mimetica','Reimeria sinuata','Reimeria sinuata f. antiqua','Reimeria uniseriata','Rexlowea navicularis','Rhoicosphenia abbreviata','Rhoicosphenia californica','Rhoicosphenia stoermeri','Rossithidium anastasiae','Rossithidium kriegeri','Rossithidium nodosum','Rossithidium petersenii','Rossithidium pusillum','Scoliopleura peisonis','Sellaphora alastos','Sellaphora americana','Sellaphora atomoides','Sellaphora auldreekie','Sellaphora bacillm','Sellaphora bacilloides','Sellaphora bacillum','Sellaphora californica','Sellaphora disjuncta','Sellaphora fusticulus','Sellaphora hohnii','Sellaphora japonica','Sellaphora javanica','Sellaphora laevissima','Sellaphora meridionalis','Sellaphora moesta','Sellaphora nigri','Sellaphora pulchra','Sellaphora pupula','Sellaphora rexii','Sellaphora saugerresii','Sellaphora stauroneioides','Sellaphora stroemii','Sellaphora subbacillum','Sellaphora subfasciata','Sellaphora wallacei','Sellaphora wangii','Semiorbis catillifera','Semiorbis eliasiae','Semiorbis rotundus','Simonsenia delognei','Skabitschewskia oestrupii','Skabitschewskia peragalloi','Skeletonema costatum','Skeletonema marinoi','Skeletonema pseudocostatum','Spicaticribra kingstonii','Stauroforma exiguiformis','Stauroneis absaroka','Stauroneis acidoclinata','Stauroneis acidoclinatopsis','Stauroneis acuta','Stauroneis agrestis','Stauroneis akamina','Stauroneis americana','Stauroneis amphicephala','Stauroneis anceps','Stauroneis ancepsfallax','Stauroneis angustilancea','Stauroneis baconiana','Stauroneis beeskovea','Stauroneis bovbjergii','Stauroneis boyntoniae','Stauroneis bryocola','Stauroneis circumborealis','Stauroneis conspicua','Stauroneis finlandia','Stauroneis fluminopsis','Stauroneis gracilis','Stauroneis heinii','Stauroneis kingstonii','Stauroneis kishinena','Stauroneis kootenai','Stauroneis kriegeri','Stauroneis lauenburgiana','Stauroneis livingstonii','Stauroneis neohyalina','Stauroneis pax','Stauroneis phoenicenteron','Stauroneis pikuni','Stauroneis pseudagrestis','Stauroneis reichardtii','Stauroneis rex','Stauroneis sacajaweae','Stauroneis schroederi','Stauroneis separanda','Stauroneis siberica','Stauroneis smithii','Stauroneis smithii var. incisa','Stauroneis sonyae','Stauroneis staurolineata','Stauroneis stodderi','Stauroneis subborealis','Stauroneis submarginalis','Stauroneis supergracilis','Stauroneis superkuelbsii','Stauroneis thompsonii','Stauroneis vandevijveri','Staurophora amphioxys','Staurophora brantii','Staurophora columbiana','Staurophora soodensis','Staurophora tackei','Staurophora wislouchii','Staurosira binodis','Staurosira construens','Staurosira construens var. venter','Staurosira stevensonii','Staurosirella berolinensis','Staurosirella leptostauron','Staurosirella leptostauron var. dubia','Staurosirella martyi','Staurosirella pinnata','Staurosirella rhomboides','Stephanodiscus alpinus','Stephanodiscus binderanus','Stephanodiscus hantzschii','Stephanodiscus hantzschii fo. tenuis','Stephanodiscus minutulus','Stephanodiscus niagarae','Stephanodiscus oregonicus','Stephanodiscus parvus','Stephanodiscus reimeri','Stephanodiscus yellowstonensis','Surirella amphioxys','Surirella angusta','Surirella arctica','Surirella atomus','Surirella brebissonii','Surirella cruciata','Surirella crumena','Surirella iowensis','Surirella lacrimula','Surirella librile','Surirella ovalis','Surirella pinnata','Surirella stalagma','Surirella striatula','Surirella suecica','Surirella tenera','Surirella terryi','Surirella undulata','Synedra cyclopum','Synedra famelica','Synedra goulardii','Synedra mazamaensis','Tabellaria fenestrata','Tabellaria flocculosa','Tabularia fasciculata','Terpsinoe musica','Tetracyclus glans','Tetracyclus hinziae','Tetracyclus rupestris','Thalassiosira baltica','Thalassiosira eccentrica','Thalassiosira lacustris','Thalassiosira minima','Thalassiosira pseudonana','Thalassiosira weissflogii','Tryblionella apiculata','Tryblionella brunoi','Tryblionella calida','Tryblionella gracilis','Tryblionella granulata','Tryblionella hungarica','Tryblionella lanceola','Ulnaria acus','Ulnaria capitata','Ulnaria contracta','Ulnaria delicatissima','Urosolenia eriensis']
sets = ["train", "test"]

wd = getcwd()
for se in sets:
    list_file = open('cls_' + se + '.txt', 'w')

    datasets_path = "datasets/" + se
    types_name = os.listdir(datasets_path)
    for type_name in types_name:
        if type_name not in classes:
            continue
        cls_id = classes.index(type_name)
        
        photos_path = os.path.join(datasets_path, type_name)
        photos_name = os.listdir(photos_path)
        for photo_name in photos_name:
            _, postfix = os.path.splitext(photo_name)
            if postfix not in ['.jpg', '.png', '.jpeg']:
                continue
            list_file.write(str(cls_id) + ";" + '%s/%s'%(wd, os.path.join(photos_path, photo_name)))
            list_file.write('\n')
    list_file.close()

