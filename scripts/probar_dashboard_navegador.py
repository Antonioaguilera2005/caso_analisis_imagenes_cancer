"""QA opcional con Chrome instalado y servidor local ya iniciado en puerto 8000."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

REPO=Path(__file__).resolve().parents[1]
SALIDA=REPO/'artifacts/qa_dashboard'


def main():
    SALIDA.mkdir(parents=True,exist_ok=True)
    errores=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path='C:/Program Files/Google/Chrome/Application/chrome.exe',headless=True)
        page=browser.new_page(viewport={'width':1500,'height':1050},device_scale_factor=1)
        page.on('pageerror',lambda e:errores.append(str(e)))
        page.on('console',lambda m:errores.append(m.text) if m.type=='error' else None)
        r=page.goto('http://127.0.0.1:8000',wait_until='networkidle')
        assert r.status==200
        page.wait_for_function("document.getElementById('connection').textContent.includes('verificado')")
        page.screenshot(path=str(SALIDA/'dashboard_desktop.png'),full_page=True)
        page.locator('#predict-button').click()
        page.wait_for_selector('#result-content:not([hidden])',timeout=90000)
        assert page.locator('#result-prob').inner_text()!='—'
        page.screenshot(path=str(SALIDA/'dashboard_resultado.png'),full_page=True)
        for section in ['resultados','entrenamiento','proyecto']:
            page.locator(f'nav a[data-view="{section}"]').click()
            assert page.locator(f'#{section}').is_visible()
            page.screenshot(path=str(SALIDA/f'dashboard_{section}.png'),full_page=True)
        page.locator('#zoom-loss').is_visible() # presente aunque se encuentre en otra sección
        page.locator('nav a[data-view="entrenamiento"]').click()
        page.locator('#fold-select').select_option('3')
        page.locator('#zoom-loss').check()
        assert 'brecha importante' in page.locator('#curve-insight').inner_text()
        page.locator('nav a[data-view="analisis"]').click()
        page.locator('[data-mode="corte"]').click()
        page.locator('#predict-button').click()
        assert 'Falta la fase PRE' in page.locator('#input-error').inner_text()
        meta=page.request.get('http://127.0.0.1:8000/api/modelo').json()
        ejemplo=meta['ejemplos'][0]
        nombres=ejemplo['archivos'][:3]
        for nombre in nombres:
            fase=nombre.rsplit('_',1)[1].split('.')[0].lower()
            page.locator(f'#file-{fase}').set_input_files(str(REPO/'artifacts/dashboard/ejemplos'/ejemplo['id']/nombre))
        page.locator('#predict-button').click()
        page.wait_for_selector('#result-content:not([hidden])',timeout=90000)
        assert page.locator('#result-slices').inner_text()=='1'
        page.locator('[data-mode="paciente"]').click()
        page.locator('#batch-files').set_input_files([str(REPO/'artifacts/dashboard/ejemplos'/ejemplo['id']/n) for n in nombres])
        page.locator('#predict-button').click()
        page.wait_for_selector('#result-content:not([hidden])',timeout=90000)
        assert '3 análisis' in page.locator('#session-count').inner_text()
        page.set_viewport_size({'width':390,'height':844})
        page.screenshot(path=str(SALIDA/'dashboard_mobile.png'),full_page=True)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert not errores,errores
        browser.close()
    (SALIDA/'verificacion.json').write_text(json.dumps({'correcto':True,'errores_js':errores,'flujo':'Ejemplo, corte individual, lote, navegación, detalle de curvas y móvil'},indent=2),encoding='utf-8')
    print('QA navegador correcto; capturas en artifacts/qa_dashboard')


if __name__=='__main__':
    main()
