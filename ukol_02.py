"""
Úkol 2
Tentokrát budeme pracovat s daty z výzkumu veřejného mínění Eurobarometr. Ten je realizován v cca 40 zemích a lidé jsou v něm pravidelně tázání na řadu otázek. My se podíváme na tři z nich.

Přehled států EU, které jsou v průzkumu, najdeš v souboru countries.csv. V první části úkolu můžeš pracovat se všemi státy nebo jen pro státy EU - záleží na tobě. Ve druhé a třetí části pracuj pouze se státy EU.

Inflace
V souboru ukol_02_a.csv najdeš procenta lidí, kteří považují inflaci a růst životních nákladů za jeden ze svých nejzávažnějších problémů. Data jsou za dvě období - léto 2022 (sloupec 97) a zima 2022/2023 (sloupec 98). Ověř, zda se procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy, změnilo.

Je vhodné provést následující postup:
"""


import pandas as pd
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn
import numpy
import statsmodels.api as sm
import statsmodels.formula.api as smf


countriesEU = pd.read_csv('countries.csv')
data = pd.read_csv('ukol_02_a.csv')
print(countriesEU.head())
print(data.head())

"""
Test normality obou skupin dat. Podle toho zjistíš, zda je lepší provést parametrický nebo neparametrický test.
"""
#Nulová hypotéza: Hodnoty mají normální rozdělení.
#Alternativní hypotéza: Hodnoty nemají normální rozložení.

res_98 = st.shapiro(data["98"])
print(res_98)
res_97 = st.shapiro(data["97"])
print(res_97)

#ShapiroResult(statistic=0.9803104996681213, pvalue=0.687289297580719)
#ShapiroResult(statistic=0.9694532752037048, pvalue=0.33090925216674805)
#Vzhledem k tomu, že pvalue obou sloupců je vyšší než 0,05, na hladině významnosti 0,05 nemáme dostatečné důkazy k zamítnutí nulové hypotézy, tedy můžeme předpokládt, že data jsou v normálním rozložení a použít parametrických testů.

"""
Formulace hypotéz testu.
Výběr vhodného testu. Vhodný je test, který jsme na lekci nepoužívali, ale je v seznamu testů, který je součástí lekce 6. Důležité je uvědomit si, že porovnáváme tu samou skupinu států ve dvou různých časových obdobích.
Formulace výsledek testu (na základě p-hodnoty).
"""

# Nulová hypotéza: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy se nezměnilo.
# Alternativní hypotéza: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy změnilo

data_frame = pd.DataFrame(data, columns=["98", "97"])
resttestrel = st.ttest_rel(data_frame["98"], data_frame["97"])
print(f"Toto je výsledek párového t-testu {resttestrel}")

data["98"].plot.kde(label='98')
data["97"].plot.kde(label='97')
plt.legend()
plt.show()

#TtestResult(statistic=-3.868878598419143, pvalue=0.0003938172257904746, df=40)
# Na základě výsledku t-testu můžeme konstatovat na hladině významnosti 0,05, že nulová hypotéza není pravdivá. Hodnota p-value je menší než 0,05, což znamená, že existuje statisticky významný rozdíl mezi procenty lidí, kteří řadí inflaci mezi své nejzávažnější problémy v obdobích léto 2022 a zima 2022/2023. Dle grafu jich přibylo.

# Další testy:

# Mann–Whitney test. Test je neparametrický, tj. nevyžaduje normální rozdělení.
resmannwhitneyu = st.mannwhitneyu(data["98"], data["97"])
print(resmannwhitneyu)
#MannwhitneyuResult(statistic=564.0, pvalue=0.010360337198200469)

#Test založený na Pearsonově korelačním koeficientu. Test je parametrický.

respearsonr = st.pearsonr(data["98"], data["97"])
print(respearsonr)
#PearsonRResult(statistic=0.5207076309684883, pvalue=0.00048264955009708245)

#  Test s využitím Spearmanova koeficientu. test je neparametrický.

resspearmanr = st.spearmanr(data["98"], data["97"])
print(resspearmanr)
#SignificanceResult(statistic=0.49765987703979275, pvalue=0.0009308002977050622)

# Test s využitím Kendallova tau. Test je neparametrický.
reskendalltau = st.kendalltau(data["98"], data["97"])
print(reskendalltau)
#SignificanceResult(statistic=0.36260011528029334, pvalue=0.001286971968439521)

#Všechny testy potvrzují výše uvedený závěr, tedy, že nulová hypotéza není pravdivá.

# Nulová hypotéza: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy výrazně narostlo.
# Alternativní hypotéza: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy výrazně nenarostlo.

ttestgreater = st.ttest_rel(data["98"], data["97"], alternative='greater')
print(ttestgreater)

#TtestResult(statistic=-3.868878598419143, pvalue=0.9998030913871048, df=40)

# Na základě tohoto testu můžeme na hladině významnosti 0,05 konstatovat, že není dostatečný důkaz o výraznějším statistickém nárůstu procenta lidí, kteří řadí inflaci mezi své nejzávažnější problémy.


merged_data = data.merge(countriesEU, on=None, left_on='Country', right_on='Country', how='inner')
print(merged_data)

# Test normality pouze z EU zemí
#Nulová hypotéza: Hodnoty mají normální rozdělení.
#Alternativní hypotéza: Hodnoty nemají normální rozložení.

res_merged98 = st.shapiro(merged_data["98"])
print(res_merged98)
res_merged97 = st.shapiro(merged_data["97"])
print(res_merged97)

#ShapiroResult(statistic=0.9399222135543823, pvalue=0.12131019681692123)
#ShapiroResult(statistic=0.952153205871582, pvalue=0.24169494211673737)
#Opět můžeme konstatovat, že vzhledem k tomu, že pvalue obou sloupců je vyšší než 0,05, na hladině významnosti 0,05 nemáme dostatečné důkazy k zamítnutí nulové hypotézy, tedy můžeme předpokládt, že data jsou v normálním rozložení a použít parametrických testů.

# Nulová hypotéza: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy se nezměnilo.
# Alternativní hypotéza: Procento lidí, kteří řadí inflaci mezi 2 své nejzávažnější problémy změnilo

data_frameEU = pd.DataFrame(merged_data, columns=["98", "97"])
resttestrelEU = st.ttest_rel(data_frameEU["98"], data_frameEU["97"])
print(f"Toto je výsledek párového t-testu pro země EU {resttestrelEU}")

merged_data["98"].plot.kde(label='98')
merged_data["97"].plot.kde(label='97')
plt.legend()
plt.show()

#TtestResult(statistic=-3.4869444202944764, pvalue=0.0017533857526091583, df=26)
# Opět můžeme konstatovat na základě výsledku t-testu na hladině významnosti 0,05, že nulová hypotéza není pravdivá. Hodnota p-value je menší než 0,05, což znamená, že existuje statisticky významný rozdíl mezi procenty lidí, kteří řadí inflaci mezi své nejzávažnější problémy v obdobích léto 2022 a zima 2022/2023. Dle grafu jich přibylo.

"""
Důvěra ve stát a v EU
Ve výzkumu je dále zkoumáno, jak moc lidé věří své národní vládě a jak moc věří EU. Data jsou v souboru ukol_02_b.csv. Číslo udává procento lidí, kteří dané instituci věří. Ověř, zda existuje korelace mezi procentem lidí, které věří EU a procentem lidí, kteří věří své národní vládě.

Je vhodné provést následující postup:

Test normality obou skupin dat.
Formulace hypotéz testu.
Volba vhodného testu. Pokud data nemají normální rozdělení, můžeš využít test korelace, který jsme prováděli na lekci. Pokud data normální rozdělení mají, můžeš použít test zmíněný v přehledu testů v dané lekci.
"""

#Test normality obou skupin dat

#Nulová hypotéza: Hodnoty mají normální rozdělení.
#Alternativní hypotéza: Hodnoty nemají normální rozložení.

duvera = pd.read_csv('ukol_02_b.csv')
print(duvera.head())

merged_duvera = duvera.merge(countriesEU, on=None, left_on='Country', right_on='Country', how='inner')
print(merged_duvera)

res_ngt = st.shapiro(merged_duvera['National Government Trust'])
print(f"Test normality duvera narodni vlady {res_ngt}")
res_eut = st.shapiro(merged_duvera['EU Trust'])
print(f"Test normality duvera EU {res_eut}")

#Test normality duvera narodni vlady ShapiroResult(statistic=0.9438267350196838, pvalue=0.15140558779239655)
#Test normality duvera EU ShapiroResult(statistic=0.9735807180404663, pvalue=0.6981646418571472)   
#Vzhledem k tomu, že pvalue obou sloupců je vyšší než 0,05, na hladině významnosti 0,05 nemáme dostatečné důkazy k zamítnutí nulové hypotézy, tedy můžeme předpokládt, že data jsou v normálním rozložení a použít parametrických testů.

# Nulová hypotéza: Mezi procentem lidí, které věří EU a procentem lidí, kteří věří své národní vládě není statistická závislost.
# Alternativní hypotéza: Mezi procentem lidí, které věří EU a procentem lidí, kteří věří své národní vládě je statistická závislost.


resduvera = st.pearsonr(merged_duvera['National Government Trust'], merged_duvera['EU Trust'])
print(resduvera)

#PearsonRResult(statistic=0.6097186340024556, pvalue=0.0007345896228823406)
# Na základě výsledku  Pearsonově korelačním koeficientu, který je nižší než hladina významnosti 0,05, zamítáme nulovou hypotézu a můžeme konstatovat, že mezi procentem lidí, které věří EU a procentem lidí, kteří věří své národní vládě je statisticky významná závislost.

"""
Důvěra v EU a euro
Nakonec si rozdělíme státy EU na dvě skupiny - státy v eurozóně a státy mimo ni. Jak je to s důvěrou v EU? Důvěřují EU více lidé, kteří žijí ve státech platící eurem? Využij znovu data o důvěře v EU ze souboru ukol_02_b.csv a rozděl státy na ty, které jsou v eurozóně, a ty, které jsou mimo ni. Porovnej, jak se liší důvěra v EU v těchto dvou skupinách zemí. Státy můžeš rozdělit s využitím tabulky v souboru countries.csv.

Test normality můžeš vynechat, řiď se výsledkem z předchozí části.
Formulace hypotéz testu.
Volba vhodného testu. Pokud data nemají normální rozdělení, můžeš využít test z bonusového úkolu ze 7 lekce. Pokud data normální rozdělení mají, můžeš použít test zmíněný v přehledu testů v dané lekci.
"""

eur_yes = merged_duvera[merged_duvera['Euro'] == 1]
print(eur_yes)
eur_no = merged_duvera[merged_duvera['Euro'] == 0]
print(eur_no)


eur_yes = merged_duvera[merged_duvera['Euro'] == 1]
eur_no = merged_duvera[merged_duvera['Euro'] == 0]

# Nulová hypotéza: Neexistuje rozdíl v důvěře v EU mezi lidmi v eurozonu a lidmi mimo eurozonu.
# Alternativní hypotéza: Existuje rozdíl v důvěře v EU mezi lidmi v eurozonu a lidmi mimo eurozonu.

print(st.ttest_ind(eur_yes['EU Trust'], eur_no['EU Trust']))

# P-hodnota je vyšší než hladina významnosti 0,05 a tedy nulovou hypotézu nezamítáme a můžeme konstatovat, že nemáme statisticky významný důkaz pro tvrzení, že lidé v eurozóně věří EU více než lidé mimo eurozónu.
