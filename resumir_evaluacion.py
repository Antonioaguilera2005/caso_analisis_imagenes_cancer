"""Informe a partir de predicciones ya guardadas: no ejecuta inferencia ni ajusta el modelo."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, auc

from analizar_resultados import casos_fallados, guardar_figura, metricas
from entrenar import uc, guardar_json
from finalizar import REPO, leer_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modelo", type=Path, default=Path("runs/modelo_final"))
    p.add_argument("--auditoria", type=Path, default=Path("runs/auditoria_final"))
    p.add_argument("--test", type=Path, default=Path("runs/test_final"))
    p.add_argument("--cv", type=Path, default=Path("cv5_v2"))
    p.add_argument("--datos", type=Path, default=Path("breastdcedl"))
    p.add_argument("--salida", type=Path, default=Path("runs/informe_final"))
    args = p.parse_args()
    if args.salida.exists():
        p.error("Usar una carpeta de informe nueva")
    a = leer_json(args.auditoria / "auditoria.json")
    if not a["correcto"]:
        p.error("La auditoría no está superada")
    registro = leer_json(REPO / "runs/test_una_vez.json")
    if registro["estado"] != "completada":
        p.error("La evaluación única de test no está completada")
    receta = leer_json(args.modelo / "receta.json")
    almacenadas = leer_json(args.test / "metricas_test.json")
    if a["sha256_checkpoint_final"] != almacenadas["sha256_checkpoint"] or registro["sha256_checkpoint"] != almacenadas["sha256_checkpoint"]:
        p.error("Los pesos auditados y evaluados son distintos")
    if registro["umbral"] != receta["umbral"] or almacenadas["umbral"] != receta["umbral"]:
        p.error("Se ha cambiado el umbral")
    train = pd.read_csv(args.auditoria / "predicciones_pacientes_train_final.csv")
    test = pd.read_csv(args.test / "predicciones_pacientes.csv")
    samples = uc.cargar_samples(args.datos)
    for nombre, tabla in (("train", train), ("test", test)):
        esperado = samples[samples.split == nombre].groupby("patient_id").pCR.first().sort_index()
        actual = tabla.set_index("patient_id").pCR.sort_index()
        if tabla.patient_id.duplicated().any() or not actual.equals(esperado):
            p.error(f"Predicciones {nombre} incompletas o etiquetas incoherentes")
        if not np.isfinite(tabla.prob).all() or not tabla.prob.between(0,1).all():
            p.error("Probabilidades no válidas")
    cortes = pd.read_csv(args.test / "predicciones_cortes.csv")
    if cortes.sample_id.duplicated().any() or set(cortes.sample_id) != set(samples[samples.split == "test"].sample_id):
        p.error("Cortes de test incompletos")
    medias = cortes.groupby("patient_id").prob.mean().sort_index()
    if not np.allclose(medias.values,test.set_index("patient_id").prob.sort_index().values,atol=1e-7):
        p.error("Predicción por paciente distinta de la media de cortes")
    mt = metricas(train.pCR, train.prob, receta["umbral"])
    me = metricas(test.pCR, test.prob, receta["umbral"])
    for clave in ("auc","accuracy","sensibilidad","especificidad","f1_macro","brier"):
        if not np.isclose(me[clave],almacenadas[clave],atol=1e-10):
            p.error(f"Métrica de test no reproducida desde CSV: {clave}")
    reporte = classification_report(test.pCR,(test.prob >= receta["umbral"]).astype(int),labels=[0,1],output_dict=True,zero_division=0)
    if not np.isclose(reporte["macro avg"]["f1-score"],me["f1_macro"]) or not np.isclose(roc_auc_score(test.pCR,test.prob),me["auc"]):
        p.error("F1 o AUC no coinciden con la comprobación independiente")
    ap_test=float(average_precision_score(test.pCR,test.prob))
    if not np.isclose(ap_test,almacenadas["average_precision"],atol=1e-10):
        p.error("Average precision no coincide con las predicciones guardadas")
    precision_pr,recall_pr,_=precision_recall_curve(test.pCR,test.prob)
    pr_auc_test=float(auc(recall_pr,precision_pr))
    pliegues = []
    for fold in range(5):
        carpeta = args.cv/f"fold{fold}_semilla42"/receta["perdida"]
        historia = leer_json(carpeta/"historial.json")
        elegida = leer_json(carpeta/"seleccion.json")["epoca"]
        h = next(h for h in historia if h["epoca"] == elegida)
        fija = next(h for h in historia if h["epoca"] == receta["epocas"])
        pliegues.append(dict(fold=fold,epoca=elegida,auc_train=h["auc_train"],auc_val=h["auc"],brecha=h["auc_train"]-h["auc"],
                             epoca_fija=receta["epocas"],auc_train_epoca_fija=fija["auc_train"],auc_val_epoca_fija=fija["auc"]))
    cv = pd.DataFrame(pliegues)
    media, desviacion = float(cv.auc_val.mean()), float(cv.auc_val.std(ddof=1))
    media_fija, desviacion_fija = float(cv.auc_val_epoca_fija.mean()), float(cv.auc_val_epoca_fija.std(ddof=1))
    baseline = metricas(test.pCR,np.zeros(len(test)),0.5)
    brier_prevalencia = float(np.mean((test.pCR-float(train.pCR.mean()))**2))
    inferior, superior = almacenadas["auc_ic95_bootstrap"]
    args.salida.mkdir(parents=True)
    cv.to_csv(args.salida/"sobreajuste_por_fold.csv",index=False)
    filas = []
    for nombre,m in (("train_final",mt),("test_final",me),("baseline_test_siempre_no_pcr",baseline)):
        filas.append(dict(conjunto=nombre,**{k:m[k] for k in ("pacientes","umbral","auc","accuracy","sensibilidad","especificidad","precision_pcr","f1_macro","f1_pcr","f1_no_pcr","brier")}))
    pd.DataFrame(filas).to_csv(args.salida/"comparacion_metricas.csv",index=False)
    guardar_json(args.salida/"verificacion_metricas.json",dict(correcto=True,classification_report=reporte,
                 auc_train=mt["auc"],auc_cv_media=media,auc_cv_std=desviacion,auc_test=me["auc"],
                 auc_cv_epoca_fija_media=media_fija,auc_cv_epoca_fija_std=desviacion_fija,
                 gap_auc_train_test=mt["auc"]-me["auc"],brier_prior_train_en_test=brier_prevalencia,
                 average_precision_test=ap_test,pr_auc_test=pr_auc_test,
                 sha256_checkpoint=almacenadas["sha256_checkpoint"],umbral=receta["umbral"]))
    fig, ax = plt.subplots(1,2,figsize=(12,4.6))
    valores = [mt["auc"],media,media_fija,me["auc"]]
    ax[0].bar(range(4),valores,color=["#2474b5","#e78b28","#b79b65","#4e8c65"])
    ax[0].errorbar(1,media,yerr=desviacion,fmt="none",color="black",capsize=5)
    ax[0].errorbar(2,media_fija,yerr=desviacion_fija,fmt="none",color="black",capsize=5)
    ax[0].errorbar(3,me["auc"],yerr=[[me["auc"]-inferior],[superior-me["auc"]]],fmt="none",color="black",capsize=5)
    for i,v in enumerate(valores):
        ax[0].text(i,v/2,f"{v:.3f}",ha="center",color="white",fontsize=12)
    ax[0].axhline(0.5,color="gray",linestyle=":")
    ax[0].set(xticks=range(4),xticklabels=["Train\nfinal","CV mejor\npor fold",f"CV época\n{receta['epocas']}","Test\nfinal"],ylim=(0,1),ylabel="ROC-AUC por paciente",
              title="CV: ± desviación entre folds\nTest: IC95% bootstrap de pacientes")
    claves=["accuracy","balanced_accuracy","sensibilidad","especificidad","f1_macro"]
    posiciones=np.arange(len(claves))
    ax[1].bar(posiciones-0.18,[me[c] for c in claves],0.36,label=f"Modelo, t={receta['umbral']:.2f}")
    ax[1].bar(posiciones+0.18,[baseline[c] for c in claves],0.36,label="Siempre no pCR")
    ax[1].set(xticks=posiciones,xticklabels=["Accuracy","Acc.\nequilibrada","Recall\npCR","Especif.","F1 macro"],ylim=(0,1),title="Test completo y umbral congelado")
    ax[1].legend(fontsize=8)
    guardar_figura(fig,args.salida/"comparacion_train_cv_test.png")
    fig,ax=plt.subplots(figsize=(8,4))
    for clase,color in ((0,"#2474b5"),(1,"#dd6c2f")):
        ax.hist(test.loc[test.pCR==clase,"prob"],bins=np.linspace(0,1,26),alpha=0.65,color=color,label=f"Real pCR={clase}")
    ax.axvline(receta["umbral"],color="black",linestyle="--",label=f"Umbral fijo {receta['umbral']:.2f}")
    ax.set(xlabel="Probabilidad media por paciente",ylabel="Pacientes",title="Test · distribución de probabilidades")
    ax.legend()
    guardar_figura(fig,args.salida/"distribucion_test.png")
    filas_cohorte=[]
    for nombre,grupo in test.groupby("dataset"):
        m=metricas(grupo.pCR,grupo.prob,receta["umbral"]) if grupo.pCR.nunique()==2 else None
        filas_cohorte.append(dict(cohorte=nombre,pacientes=len(grupo),pcr_positivas=int(grupo.pCR.sum()),
                                  auc=m["auc"] if m else None,sensibilidad=m["sensibilidad"] if m else None,
                                  especificidad=m["especificidad"] if m else None,f1_macro=m["f1_macro"] if m else None))
    cohortes=pd.DataFrame(filas_cohorte)
    cohortes.to_csv(args.salida/"metricas_por_cohorte.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.bar(cohortes.cohorte,cohortes.auc,color=["#2474b5","#dd6c2f","#4e8c65"][:len(cohortes)])
    for i,c in enumerate(filas_cohorte):
        if c["auc"] is not None:
            ax.text(i,c["auc"]+0.025,f"AUC={c['auc']:.3f}\nn={c['pacientes']}; pCR={c['pcr_positivas']}",ha="center",fontsize=9)
    ax.axhline(0.5,color="gray",linestyle=":")
    ax.set(ylim=(0,1),ylabel="ROC-AUC por paciente",title="Test por cohorte · tamaños pequeños e incertidumbre\nSin cambiar el modelo ni invertir scores a partir de test")
    guardar_figura(fig,args.salida/"auc_test_por_cohorte.png")
    casos_fallados(test,"test_final",samples[samples.split=="test"],args.datos,args.salida,umbral=receta["umbral"])
    artefacto=leer_json(args.modelo/"artefacto.json")
    texto=["# Auditoría y evaluación final", "", "El informe lee predicciones ya guardadas; no repite inferencia ni busca nuevos umbrales.","",
           f"Entrenamiento completo: {artefacto['segundos']:.1f} s; cinco épocas fijadas previamente por CV.",
           f"Receta: CNN de {receta['parametros']} parámetros, pérdida {receta['perdida']}, umbral {receta['umbral']:.2f} y media de cortes.","",
           "## Resultados por paciente", "",f"Train final: AUC {mt['auc']:.3f} en {len(train)} pacientes (no es validación).",
           f"CV seleccionando la mejor época en cada fold: AUC media {media:.3f} ± {desviacion:.3f}.",
           f"CV en la época fija {receta['epocas']}: AUC media {media_fija:.3f} ± {desviacion_fija:.3f}. No son la misma evaluación.",
           f"Test: AUC {me['auc']:.3f}, IC95% [{inferior:.3f}, {superior:.3f}], en {len(test)} pacientes.",
           f"Test: accuracy {me['accuracy']:.3f}, sensibilidad {me['sensibilidad']:.3f}, especificidad {me['especificidad']:.3f}, F1 macro {me['f1_macro']:.3f}.",
           f"Precision pCR={me['precision_pcr']}; AP={ap_test:.3f}; PR-AUC trapezoidal={pr_auc_test:.3f}.",
           f"Baseline siempre no pCR: accuracy {baseline['accuracy']:.3f}, F1 macro {baseline['f1_macro']:.3f}.","",
           "## Comprobaciones y límites", "", "Se verificaron los cinco pesos y pasos de Adam, la partición completa y la aritmética de los historiales.",
           "El modelo definitivo se recalculó sobre todo train y se contrastó con el historial GPU. AUC y F1 de test se contrastaron desde CSV con scikit-learn.",
           "Una curva de train por sí sola no demuestra ausencia de sobreajuste. La evaluación independiente de test aporta evidencia para este conjunto, no garantía clínica.",
           "Los modelos de CV y el modelo final son distintos; sus métricas no forman una curva de validación del mismo reentrenamiento.",
           "La CV con selección del máximo en cada fold puede ser optimista. La receta de mediana de épocas no hereda automáticamente ese AUC; se muestra también la época fija.",
           "El número máximo de épocas de otros trabajos no es comparable con las épocas elegidas mediante CV y sus tiempos por época.",
           "No se ajustaron pérdida, épocas o umbral con test. A partir de aquí no se reentrena ni se busca otro umbral usando estos resultados.",
           "La escala de detalle de BCE amplifica cambios pequeños: consultar también la escala con contexto de la auditoría.",
           "Brier se compara también con una probabilidad constante igual a la prevalencia de train; ponderar una pérdida no garantiza calibración.",
           f"Brier test modelo={me['brier']:.4f}; baseline prevalencia train={brier_prevalencia:.4f}.",
           "Modelos, métricas, imágenes de errores y este informe permanecen fuera de Git. Uso exclusivamente educativo."]
    texto.extend(["","## Heterogeneidad entre cohortes",""])
    for c in filas_cohorte:
        texto.append(f"{c['cohorte']}: n={c['pacientes']}, pCR positivas={c['pcr_positivas']}, AUC={c['auc']}, sensibilidad={c['sensibilidad']}.")
    texto.extend(["","Los subgrupos pequeños no permiten conclusiones clínicas firmes. Un mal resultado en una cohorte no debe ocultarse detrás de la media global.",
                  "La diferencia global train–test es descriptiva y no prueba ausencia de sobreajuste, calibración o generalización fuera de estas cohortes."])
    (args.salida/"INFORME_FINAL.md").write_text("\n".join(texto)+"\n",encoding="utf-8")
    print(f"Informe y PNG generados desde predicciones guardadas: {args.salida}")


if __name__=="__main__":
    main()
