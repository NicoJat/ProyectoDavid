import pandas as pd
import numpy as np
from RBSConnection import RBSConnection
from EndPlate4EConnection import EndPlate4EConnection
from EndPlate4ESConnection import EndPlate4ESConnection
from Diafragma_externo import DiafragmaExterno
from WUFConnection import WUFConnection

class StructuralDesign:
    def __init__(self, materiales, datos_arquitectonicos, propiedades_vigas, propiedades_columna, propiedades_conexiones, parametros_conexiones):
        # Inicializar parámetros
        self.materiales = materiales
        self.datos_arq = datos_arquitectonicos
        self.prop_vigas = propiedades_vigas
        self.prop_col = propiedades_columna
        self.prop_conexiones = propiedades_conexiones
        self.param_conex = parametros_conexiones
        
        # Inicializar resultados
        self.propiedades_vigas_sec = {}
        self.propiedades_conexiones = {}
        self.propiedades_vigas_trab = {}
        self.data_vigas_sec = {}
        self.data_vigas_trab = {}
        self.data_columna = {}
        self.data_conexiones = {}
        self.resultados_RBS = {}
        self.resultados_end_plate_4E = {}
        self.resultados_end_plate_4ES = {}
        self.resultados_WUF_W = {}
        self.resultados_Diafragma_externo = {}

    def _calcular_propiedades_iniciales(self):
        """Calcula propiedades básicas de vigas y columnas"""
        # Propiedades de vigas secundarias
        for lado, viga in self.prop_vigas['vigas_sec_prop'].items():
            self.propiedades_vigas_sec[lado] = self._calcular_propiedades( viga)
        
        # Propiedades de vigas trabe
        for lado, viga in self.prop_vigas['vigas_trab_prop'].items():
            self.propiedades_vigas_trab[lado] = self._calcular_propiedades(viga)
        
        # Diseño de vigas secundarias
        for lado in self.prop_vigas['vigas_secundarias']:
            self.data_vigas_sec[lado] = self._calcular_propiedades_secundarias(
                self.prop_vigas['vigas_secundarias'][lado],
                self.propiedades_vigas_sec[lado]
            )
        
        # Diseño de vigas trabe
        for lado in self.prop_vigas['vigas_trab']:
            self.data_vigas_trab[lado] = self._calcular_propiedades_trabe(
                lado,
                self.prop_vigas['vigas_trab'][lado],
                self.propiedades_vigas_trab[lado]
            )
        
        #Columnas
        self.data_columna['columna'] = self._calculo_columna(
            self.prop_col['columna'],
            self.prop_col['Atiesadores']
        )

        #Conexiones
        for lado in self.prop_vigas['vigas_secundarias']:
            self.data_conexiones[lado] = self._calcular_conexiones(
                self.data_vigas_sec[lado],
                self.propiedades_vigas_sec[lado],
                lado,
                self.param_conex
            )

    def _calcular_propiedades(self, viga):
        fy = viga['Acero'] * 1000 / 14.23
        E = self.materiales['E']
        bf, tf, h, tw = viga['bf'], viga['tf'], viga['h'], viga['tw']
        A = bf * h - (bf - tw) * (h - 2 * tf)
        Peso = A * 0.785
        Ry = 1.3 if viga['Acero'] == 36 else 1.1
        Ix = (bf * (h ** 3) / 12) - ((bf - tw) * (h - 2 * tf) ** 3 / 12)
        Zx = bf * tf * (h - tf) + (tw * ((h / 2) - tf) ** 2)
        cte = (E/fy)**0.5
        cf1 = cte*0.3
        cf2 = cte*0.38
        cw1 = cte*2.45
        cw2 = cte*3.76
        Condicion_bf = "SISM" if bf/2/tf < cf1 else "COMP" if bf/2/tf < cf2 else "NCOMP"
        Condicion_h = "SISM" if h/tw < cw1 else "COMP" if h/tw < cw2 else "NCOMP"

        # Define the desired order of parameters
        ordered_parameters = [
            'cte','cf1','cf2','cw1','cw2','A', 'Peso', 'Ix', 'Zx', 'h', 'Condición h', 'bf',
            'Condición bf', 'Ry', 'tf', 'tw'
        ]

        valores = {
            'cte': cte,
            'cf1':cf1,
            'cf2':cf2,
            'cw1':cw1,
            'cw2':cw2,
            'A': A,
            'Peso': Peso,
            'Ix': Ix,
            'Zx': Zx,
            'h': h,
            'Condición h': Condicion_h,
            'bf': bf,
            'Condición bf': Condicion_bf,
            'Ry': Ry,
            'tf': tf,
            'tw': tw
        }

        unidades = {
            'cte': 'cte',
            'cf1':'cf1',
            'cf2':'cf2',
            'cw1':'cw1',
            'cw2':'cw2',
            'A': 'cm²',
            'Peso': 'kg/m',
            'Ix': 'cm⁴',
            'Zx': 'cm³',
            'h': 'cm',
            'Condición h': '-',
            'bf': 'cm',
            'Condición bf': '-',
            'Ry': '-',
            'tf': 'cm',
            'tw': 'cm'
        }

        # Create ordered dictionaries for values and units
        ordered_valores = {param: valores[param] for param in ordered_parameters}
        ordered_unidades = {param: unidades[param] for param in ordered_parameters}

        return {
            'valores': ordered_valores,
            'unidades': ordered_unidades
        }

    def _calcular_propiedades_secundarias(self, viga_secundaria, propiedades_viga):
        fy = viga_secundaria['Acero'] * 1000 / 14.23
        Lv, Lt, Vigas = viga_secundaria['Lv'], viga_secundaria['Lt'], viga_secundaria['Vigas']
        Peso = propiedades_viga['valores']['Peso']
        at = Lt / (Vigas + 1)
        W = Lv * at * self.datos_arq['Cu'] + Peso / 1000 * Lv * 1.2
        Mu = W * Lv / 8
        Mr = 0.9 * propiedades_viga['valores']['Zx'] * fy / 100000
        D_C = Mu / Mr
        S_tablas = Mu * 100000 / 0.8 / propiedades_viga['valores']['h']
        unidades = {
            'fy': 'kg/cm²',
            'S_tablas': 'cm³',
            'W': 'tonf/m',
            'Mu': 'tonf·m',
            'Mr': 'tonf·m',
            'D_C': '-'
        }

        return {
            'valores': {
                'fy': fy,
                'S_tablas': S_tablas,
                'W': W,
                'Mu': Mu,
                'Mr': Mr,
                'D_C': D_C
            },
            'unidades': unidades
        }

    def _calcular_propiedades_trabe(self, lado, viga_trabe, propiedades_viga):
        fy = viga_trabe['Acero'] * 1000 / 14.23
        Lt = viga_trabe['Lt']
        cv1 = 1.1*(5*self.materiales['E']/fy)**0.5
        cv2 = 1.37*(5*self.materiales['E']/fy)**0.5
        h_tw = propiedades_viga['valores']['h'] / propiedades_viga['valores']['tw']
        if h_tw < cv1:
            cv = 1
        elif h_tw < cv2:
            cv = cv1 / h_tw
        else:
            cv = 1.51 * 5 * self.materiales['E'] / h_tw**2 / fy
        W = self.data_vigas_sec['iz']['valores']['W'] if lado == 'iz' else self.data_vigas_sec['der']['valores']['W']
        Pi = W / 2 if viga_trabe['Tipo'] == "Borde" else W / 2 * (1 + (self.datos_arq['L4'] / self.datos_arq['L3']))
        Mu = Pi * Lt * (((viga_trabe['Num'] + 1) ** 2) - 1) / 12 / (viga_trabe['Num'] + 1)
        Mr = 0.9 * propiedades_viga['valores']['Zx'] * fy / 100000
        D_C = Mu / Mr
        unidades = {
            'fy': 'kg/cm²',
            'cv1': '-',
            'cv2': '-',
            'cv': '-',
            'Mu': 'tonf/m',
            'Mr': 'tonf·m',
            'D_C': '-',
            'Pi': 'tonf'
        }

        return {
            'valores': {
                'fy': fy,
                'cv1': cv1,
                'cv2': cv2,
                'cv': cv,
                'Mu': Mu,
                'Mr': Mr,
                'D_C': D_C,
                'Pi': Pi
            },
            'unidades': unidades
        }

    def _calculo_columna(self, prop_columna, prop_atiesadores):
        b = prop_columna['b']
        h = prop_columna['h']
        e = prop_columna['e']
        A = prop_columna['Acero']
        
        Num_b = prop_atiesadores['Sentido_b']['Num']
        I_b = prop_atiesadores['Sentido_b']['I']
        e_b = prop_atiesadores['Sentido_b']['e']
        Ia_b = e*I_b**3 /12
        
        Num_h = prop_atiesadores['Sentido_h']['Num']
        I_h = prop_atiesadores['Sentido_h']['I']
        e_h = prop_atiesadores['Sentido_h']['e']
        Ia_h = e*I_h**3 /12

        As = (b*h-(b-2*e)*(h-2*e))+I_b*e*2*Num_b*2+I_h*e_h*2*Num_h*2
        
        fc = self.materiales['fc']
        E_steel = self.materiales['E']
        L1, L2, L3, L4 = self.datos_arq['L1'], self.datos_arq['L2'], self.datos_arq['L3'], self.datos_arq['L4']
        Cu, Pisos, He = self.datos_arq['Cu'], self.datos_arq['Pisos'], self.datos_arq['He']

        Ac = b*h-As
        Ec = 12600*fc**0.5
        fy = A*1000/14.23
        fym = fy+0.85*fc*Ac/As
        Em = E_steel+0.4*Ec*Ac/As
        At = (L1/2+L2/2)*(L3/2+L4/2)

        if prop_columna['Relleno hormigon'] == 'SI':
            E = Em
            fy_eff = fym
        else:
            E = E_steel
            fy_eff = fy
        Imin = 9.4*e**4
        
        Peso = As*0.785
        Ix = (b*(h**3)/12)-(b-(2*e))*((h-(2*e))**3 /12)
        Iy = (h*b**3 /12)-((h-2*e)*((b-2*2)**3 /12))
        Zx = e*h**2 /2+e*(b-2*e)*(h-e)
        Zy = e*b**2 /2+e*(h-2*e)*(b-e)
        Pu = At*Cu*Pisos/0.85
        
        rx = (Ix/As)**0.5
        ry = (Iy/As)**0.5
        esbx = 1.2*He*100/rx
        esby = 1.2*He*100/ry
        fex = np.pi**2*E/esbx**2
        fey = np.pi**2*E/esby**2
        
        Prx = 0.658**(fy_eff/fex)*fy_eff*0.9*As/1000 if esbx<4.71*(E/fy_eff)**0.5 else 0.877*fex*0.9*As/1000
        Pry = 0.658**(fy_eff/fey)*fy_eff*0.9*As/1000 if esby<4.71*(E/fy_eff)**0.5 else 0.877*fey*0.9*As/1000
        Pr = min(Prx,Pry)
        D_C = Pu/Pr
        Mp = Zy*(2*fy_eff-Pu*1000/As-(Pu*1000-Pu*1000/Pisos)/As)/100000
        unidades = {
            'Prx': 'tonf',
            'Pry': 'tonf',
            'D_C': '-',
            'Mp': 'tonf·m',
            'Imin': 'cm⁴',
            'fy': 'kg/cm²'
        }

        return {
            'valores': {
                'Prx': Prx,
                'Pry': Pry,
                'D_C': D_C,
                'Mp': Mp,
                'Imin': Imin,
                'fy': fy
            },
            'unidades': unidades
        }

    def _calcular_conexiones(self, viga_secundaria, propiedades_viga, lado, parametros_conexiones):
        W = viga_secundaria['valores']['W'] / 2
        fy = viga_secundaria['valores']['fy']

        params = parametros_conexiones[lado]

        Perno_an_ap = params['angulo_apernado']['Perno']
        hp_an_ap = params['angulo_apernado']['hp']
        Lbord_an_ap = params['angulo_apernado']['Lbord']
        bp_an_ap = params['angulo_apernado']['bp']
        dp_an_ap = Perno_an_ap * 2.54
        da_an_ap = (Perno_an_ap + 0.125) * 2.54
        Ap_an_ap = np.pi * (dp_an_ap ** 2) / 4
        Q_an_ap = 0.75 * Ap_an_ap * 3.795
        Num_an_ap = max(2, int(W / Q_an_ap) + 1)
        s_an_ap = (hp_an_ap - 2 * Lbord_an_ap) / (Num_an_ap - 1)
        tp_an_ap = max(propiedades_viga['valores']['tw'], np.ceil(W * 1000 / (Num_an_ap * 2.4 * dp_an_ap * 3795)), 1)
        Ant_an_ap = (bp_an_ap / 2 - da_an_ap / 2) * tp_an_ap
        Av_an_ap = (hp_an_ap - Lbord_an_ap) * tp_an_ap
        Anv_an_ap = (hp_an_ap - Lbord_an_ap - (Num_an_ap - 1 + 0.5) * da_an_ap) * tp_an_ap
        BC_an_ap = min(0.75 * (0.6 * 3795 * Anv_an_ap + 3795 * Ant_an_ap),
                        0.75 * (0.6 * fy * Av_an_ap + 3795 * Ant_an_ap)) / 1000
        Desg_an_ap = min(0.9 * bp_an_ap * tp_an_ap * fy / 1000,
                          0.75 * (bp_an_ap - da_an_ap) * tp_an_ap * 3795 / 1000)

        E_an_sold = params['angulo_soldado']['E']
        g_an_sold = params['angulo_soldado']['g']
        hp_an_sold = params['angulo_soldado']['hp']
        tp_an_sold = params['angulo_soldado']['tp']
        bp_an_sold = params['angulo_soldado']['bp']
        Rs_an_sold = 0.32 * E_an_sold * g_an_sold * hp_an_sold / 14.23
        Des_an_sold = 0.9 * bp_an_sold * tp_an_sold * fy / 1000

        Ip_as_ap = propiedades_viga['valores']['bf'] + 4
        Perno_as_ap = params['asiento_apernado']['Perno']
        bp_as_ap = params['asiento_apernado']['bp']
        dp_as_ap = Perno_as_ap * 2.54
        da_as_ap = (Perno_as_ap + 0.125) * 2.54
        Ap_as_ap = np.pi * dp_as_ap ** 2 / 4
        Q_as_ap = 0.75 * Ap_as_ap * 3.795
        Num_as_ap = max(2, int(W / Q_as_ap) + 1)
        tp_as_ap = max(0.3, np.ceil(W * 1000 / Num_as_ap / 2.4 / dp_as_ap / 3795), 1)
        Des_as_ap = min(0.9 * Ip_as_ap * tp_as_ap * fy / 1000,
                         0.75 * (Ip_as_ap - Num_as_ap * da_as_ap) * tp_as_ap * 3795 / 1000)

        E_as_sold = params['asiento_soldado']['E']
        Ip_as_sold = propiedades_viga['valores']['bf'] + 4
        bp_as_sold = params['asiento_soldado']['bp']
        tp_as_sold = params['asiento_soldado']['tp']
        Rs_as_sold = 0.32 * E_as_sold * tp_as_sold * 2 * bp_as_sold / 14.23

        E_al_al = params['alma_alma']['E']
        hs_al_al = params['alma_alma']['hs']
        g_al_al = params['alma_alma']['g']
        Rs_al_al = 0.32 * E_al_al * 0.6 * propiedades_viga['valores']['tw'] * hs_al_al / 14.23 * 2

        return {
            'angulo_apernado': {
                'valores': {
                    'Vu': W,
                    'Perno': Perno_an_ap,
                    'hp': hp_an_ap,
                    'Lbord': Lbord_an_ap,
                    'bp': bp_an_ap,
                    'tp': tp_an_ap,
                    'BC': BC_an_ap,
                    'Desg': Desg_an_ap
                },
                'unidades': {
                    'Vu': 'tonf',
                    'Perno': 'in',
                    'hp': 'cm',
                    'Lbord': 'cm',
                    'bp': 'cm',
                    'tp': 'cm',
                    'BC': 'tonf',
                    'Desg': 'tonf'
                }
            },
            'angulo_soldado': {
                'valores': {
                    'Vu': W,
                    'E': E_an_sold,
                    'hp': hp_an_sold,
                    'bp': bp_an_sold,
                    'tp': tp_an_sold,
                    'Rs': Rs_an_sold,
                    'Des': Des_an_sold
                },
                'unidades': {
                    'Vu': 'tonf',
                    'E': 'kg/cm²',
                    'hp': 'cm',
                    'bp': 'cm',
                    'tp': 'cm',
                    'Rs': 'tonf',
                    'Des': 'tonf'
                }
            },
            'asiento_apernado': {
                'valores': {
                    'Vu': W,
                    'Ip': Ip_as_ap,
                    'Perno': Perno_as_ap,
                    'bp': bp_as_ap,
                    'tp': tp_as_ap,
                    'Des': Des_as_ap
                },
                'unidades': {
                    'Vu': 'tonf',
                    'Ip': 'cm',
                    'Perno': 'in',
                    'bp': 'cm',
                    'tp': 'cm',
                    'Des': 'tonf'
                }
            },
            'asiento_soldado': {
                'valores': {
                    'Vu': W,
                    'E': E_as_sold,
                    'Ip': Ip_as_sold,
                    'bp': bp_as_sold,
                    'tp': tp_as_sold,
                    'Rs': Rs_as_sold
                },
                'unidades': {
                    'Vu': 'tonf',
                    'E': 'kg/cm²',
                    'Ip': 'cm',
                    'bp': 'cm',
                    'tp': 'cm',
                    'Rs': 'tonf'
                }
            },
            'alma_alma': {
                'valores': {
                    'Vu': W,
                    'E': E_al_al,
                    'hs': hs_al_al,
                    'Rs': Rs_al_al
                },
                'unidades': {
                    'Vu': 'tonf',
                    'E': 'kg/cm²',
                    'hs': 'cm',
                    'Rs': 'tonf'
                }
            }
        }
    
    # --- New methods for individual connection calculations ---
    def calculate_rbs_connection(self, lado):
        """Calculates and stores the results for RBS connection for a given side."""
        if lado not in ['iz', 'der']:
            raise ValueError("Lado must be 'iz' or 'der'.")
    
        
        rbs_conn = RBSConnection(
            self.prop_conexiones['Prop_RBS'][lado],
            self.propiedades_vigas_trab[lado],
            self.data_vigas_trab[lado],
            self.prop_vigas['vigas_trab'][lado],
            self.materiales,
            self.datos_arq
        )
        self.resultados_RBS[lado] = rbs_conn.calcular()
        return self.resultados_RBS[lado]

    def calculate_end_plate_4e_connection(self, lado):
        """Calculates and stores the results for End Plate 4E connection for a given side."""
        if lado not in ['iz', 'der']:
            raise ValueError("Lado must be 'iz' or 'der'.")

        end_plate_4e_conn = EndPlate4EConnection(
            self.prop_conexiones['Prop_end_plate_4E'][lado],
            self.propiedades_vigas_trab[lado],
            self.data_vigas_trab[lado],
            self.prop_col['columna'],
            self.datos_arq
        )
        self.resultados_end_plate_4E[lado] = end_plate_4e_conn.calcular()
        return self.resultados_end_plate_4E[lado]

    def calculate_end_plate_4es_connection(self, lado):
        """Calculates and stores the results for End Plate 4ES connection for a given side."""
        if lado not in ['iz', 'der']:
            raise ValueError("Lado must be 'iz' or 'der'.")

        end_plate_4es_conn = EndPlate4ESConnection(
            self.prop_conexiones['Prop_end_plate_4ES'][lado],
            self.propiedades_vigas_trab[lado],
            self.data_vigas_trab[lado],
            self.prop_col['columna'],
            self.data_columna,
            self.prop_vigas['vigas_trab'][lado],
            self.datos_arq
        )
        self.resultados_end_plate_4ES[lado] = end_plate_4es_conn.calcular()
        return self.resultados_end_plate_4ES[lado]
    
    def calculate_Diafragma_externo_connection(self, lado):
        """Calculates and stores the results for Diafragma externo connection for a given side."""
        if lado not in ['iz', 'der']:
            raise ValueError("Lado must be 'iz' or 'der'.")

        Diaf_extern_conn = DiafragmaExterno(
            self.prop_conexiones['Diafragma_externo'][lado],
            self.propiedades_vigas_trab[lado],
            self.data_vigas_trab[lado],
            self.prop_col['columna'],
            self.data_columna,
            self.prop_vigas['vigas_trab'][lado],
            self.datos_arq
        )
        self.resultados_Diafragma_externo[lado] = Diaf_extern_conn.calcular()
        return self.resultados_Diafragma_externo[lado]

    def calculate_wuf_w_connection(self, lado):
        """Calculates and stores the results for WUF-W connection for a given side."""
        if lado not in ['iz', 'der']:
            raise ValueError("Lado must be 'iz' or 'der'.")

        wuf_w_conn = WUFConnection(
            self.prop_conexiones['Prop_WUF_W'][lado],
            self.propiedades_vigas_trab[lado],
            self.data_vigas_trab[lado],
            self.prop_col['columna'],
            self.datos_arq
        )
        self.resultados_WUF_W[lado] = wuf_w_conn.calcular()
        return self.resultados_WUF_W[lado]

    def mostrar_resultados(self, redondeo=3):
        # Helper function to display combined results for 'iz' and 'der'
        def display_combined_results(results_dict, title, redondeo_val):
            if not results_dict:
                print(f"No hay resultados para {title}.")
                return

            print(f"\n--- {title} ---")

            is_single_item = False
            if 'columna' in results_dict and not ('iz' in results_dict['columna'] or 'der' in results_dict['columna']):
                is_single_item = True
            elif 'iz' not in results_dict and 'der' not in results_dict:
                if len(results_dict) == 1:
                    first_key = next(iter(results_dict))
                    if 'valores' in results_dict[first_key] and 'unidades' in results_dict[first_key]:
                        is_single_item = True

            if is_single_item:
                key_name = next(iter(results_dict))
                data = results_dict[key_name]
                # Ensure the order of index (parameters) is taken directly from the 'valores' keys
                df = pd.DataFrame({
                    'Valor': data['valores'],
                    'Unidad': data['unidades']
                }).round(redondeo_val)
                print(f"\n{key_name.replace('_', ' ').title()}")
                display(df)
            else:
                combined_data = []
                # Determine the primary keys (e.g., 'angulo_apernado' or just the main parameters)
                primary_keys = []
                if 'iz' in results_dict and isinstance(results_dict['iz'], dict) and any(isinstance(v, dict) and 'valores' in v for v in results_dict['iz'].values()):
                    primary_keys = sorted({k for v in results_dict.values() for k in v.keys()})
                elif 'iz' in results_dict and 'valores' in results_dict['iz']:
                    primary_keys = ['main'] # A dummy key to loop once for the main data
                elif 'der' in results_dict and 'valores' in results_dict['der']:
                    primary_keys = ['main'] # A dummy key to loop once for the main data
                
                # Get a reference for the desired parameter order (from 'iz' if available, else 'der')
                # This assumes 'iz' and 'der' will have the same parameter keys in the same order
                reference_parameters = []
                if 'iz' in results_dict:
                    if 'main' in primary_keys:
                        reference_parameters = list(results_dict['iz'].get('valores', {}).keys())
                    else:
                        # For nested structures, get parameters from the first sub-section found
                        for pk in primary_keys:
                            if pk in results_dict['iz'] and 'valores' in results_dict['iz'][pk]:
                                reference_parameters = list(results_dict['iz'][pk]['valores'].keys())
                                break
                elif 'der' in results_dict: # Fallback if 'iz' is not present
                    if 'main' in primary_keys:
                        reference_parameters = list(results_dict['der'].get('valores', {}).keys())
                    else:
                        for pk in primary_keys:
                            if pk in results_dict['der'] and 'valores' in results_dict['der'][pk]:
                                reference_parameters = list(results_dict['der'][pk]['valores'].keys())
                                break

                for pk in primary_keys:
                    # Use the reference_parameters list to maintain consistent order
                    for param in reference_parameters:
                        row = {'Sección': pk.replace('_', ' ').title() if pk != 'main' else 'General', 'Parámetro': param}

                        if pk == 'main':
                            iz_values = results_dict.get('iz', {}).get('valores', {})
                        else:
                            iz_values = results_dict.get('iz', {}).get(pk, {}).get('valores', {})
                        row['Izquierda (Valor)'] = round(iz_values.get(param, np.nan), redondeo_val) if isinstance(iz_values.get(param), (int, float)) else iz_values.get(param, np.nan)

                        if pk == 'main':
                            der_values = results_dict.get('der', {}).get('valores', {})
                            der_unit = results_dict.get('der', {}).get('unidades', {}).get(param, '-')
                        else:
                            der_values = results_dict.get('der', {}).get(pk, {}).get('valores', {})
                            der_unit = results_dict.get('der', {}).get(pk, {}).get('unidades', {}).get(param, '-')

                        row['Derecha (Valor)'] = round(der_values.get(param, np.nan), redondeo_val) if isinstance(der_values.get(param), (int, float)) else der_values.get(param, np.nan)
                        row['Unidad'] = der_unit

                        if not (pd.isna(row['Izquierda (Valor)']) and pd.isna(row['Derecha (Valor)'])):
                            combined_data.append(row)

                if combined_data:
                    df_combined = pd.DataFrame(combined_data)
                    df_combined = df_combined[['Sección', 'Parámetro', 'Izquierda (Valor)', 'Derecha (Valor)', 'Unidad']]
                    display(df_combined)
                else:
                    print(f"No hay datos detallados para mostrar para {title}.")
                    
        # Display sections using the new helper function
        display_combined_results(self.propiedades_vigas_sec, "Propiedades de las Vigas Secundarias", redondeo)
        display_combined_results(self.propiedades_vigas_trab, "Propiedades de las Vigas Trabe", redondeo)
        display_combined_results(self.data_vigas_sec, "Resultados de Vigas Secundarias", redondeo)
        display_combined_results(self.data_vigas_trab, "Resultados de Vigas Trabe", redondeo)
        display_combined_results(self.data_columna, "Diseño de Columnas", redondeo) # This will be single table if only one 'columna' entry
        display_combined_results(self.data_conexiones, "Conexiones (Vigas Secundarias)", redondeo)
        



        def display_advanced_connection_results(results_dict, connection_name, redondeo):
            if not results_dict:
                print(f"No hay resultados para conexiones {connection_name}. Ejecute el cálculo correspondiente primero.")
                return

            print(f"\n--- Resultados de Conexiones Avanzadas {connection_name} ---")
            
            all_parameters = set()
            for lado_data in results_dict.values():
                for sub_section_data in lado_data.values():
                    if 'valores' in sub_section_data:
                        all_parameters.update(sub_section_data['valores'].keys())

            combined_data = []
            for sub_section in sorted({k for v in results_dict.values() for k in v.keys()}):
                for param in sorted(list(all_parameters)):
                    row = {'Sección': sub_section.replace('_', ' ').title(), 'Parámetro': param}
                    
                    iz_values = results_dict.get('iz', {}).get(sub_section, {}).get('valores', {})
                    row['Izquierda (Valor)'] = round(iz_values.get(param, np.nan), redondeo) if isinstance(iz_values.get(param), (int, float)) else iz_values.get(param, np.nan)

                    der_values = results_dict.get('der', {}).get(sub_section, {}).get('valores', {})
                    der_unit = results_dict.get('der', {}).get(sub_section, {}).get('unidades', {}).get(param, '-')
                    row['Derecha (Valor)'] = round(der_values.get(param, np.nan), redondeo) if isinstance(der_values.get(param), (int, float)) else der_values.get(param, np.nan)
                    row['Unidad'] = der_unit
                    
                    if not (pd.isna(row['Izquierda (Valor)']) and pd.isna(row['Derecha (Valor)'])):
                        combined_data.append(row)

            if combined_data:
                df_combined = pd.DataFrame(combined_data)
                df_combined = df_combined[['Sección', 'Parámetro', 'Izquierda (Valor)', 'Derecha (Valor)', 'Unidad']]
                display(df_combined)
            else:
                print(f"No hay datos detallados para mostrar para las conexiones {connection_name}.")


        display_advanced_connection_results(self.resultados_RBS, "RBS", redondeo)
        display_advanced_connection_results(self.resultados_end_plate_4E, "End Plate 4E", redondeo)
        display_advanced_connection_results(self.resultados_end_plate_4ES, "End Plate 4ES", redondeo)
        display_advanced_connection_results(self.resultados_Diafragma_externo, "Diafragma Externo", redondeo)
        display_advanced_connection_results(self.resultados_WUF_W, "WUF-W", redondeo)


    def exportar_a_excel(self, nombre_archivo="Resultados_Diseno_Estructural.xlsx"):
        print(f"Iniciando exportación a '{nombre_archivo}'...")
        
        with pd.ExcelWriter(nombre_archivo, engine='openpyxl') as writer:
            # Helper function to export combined results
            def export_combined_data_to_excel(results_dict, sheet_title, writer):
                if not results_dict:
                    return

                sheet_name = sheet_title[:31] # Excel sheet name limit
                
                # Check if it's a single item (like 'columna') or 'iz'/'der' structure
                is_single_item = False
                if 'columna' in results_dict and not ('iz' in results_dict['columna'] or 'der' in results_dict['columna']):
                    is_single_item = True
                elif 'iz' not in results_dict and 'der' not in results_dict:
                    if len(results_dict) == 1:
                        first_key = next(iter(results_dict))
                        if 'valores' in results_dict[first_key] and 'unidades' in results_dict[first_key]:
                            is_single_item = True


                if is_single_item:
                    key_name = next(iter(results_dict))
                    data = results_dict[key_name]
                    df = pd.DataFrame({
                        'Valor': data['valores'],
                        'Unidad': data['unidades']
                    }).round(3)
                    df.to_excel(writer, sheet_name=sheet_name, index_label='Parámetro')
                else:
                    all_parameters = set()
                    if 'iz' in results_dict:
                        for sub_section_data in results_dict.get('iz', {}).values():
                            if 'valores' in sub_section_data:
                                all_parameters.update(sub_section_data['valores'].keys())
                    if 'der' in results_dict:
                        for sub_section_data in results_dict.get('der', {}).values():
                            if 'valores' in sub_section_data:
                                all_parameters.update(sub_section_data['valores'].keys())

                    # If no 'valores' keys, check top-level parameters directly
                    if not all_parameters and 'iz' in results_dict and 'valores' in results_dict['iz']:
                        all_parameters.update(results_dict['iz']['valores'].keys())
                    if not all_parameters and 'der' in results_dict and 'valores' in results_dict['der']:
                        all_parameters.update(results_dict['der']['valores'].keys())

                    combined_data = []
                    if 'iz' in results_dict and isinstance(results_dict['iz'], dict) and any(isinstance(v, dict) and 'valores' in v for v in results_dict['iz'].values()):
                        primary_keys = sorted({k for v in results_dict.values() for k in v.keys()})
                    else:
                        primary_keys = ['main']

                    for pk in primary_keys:
                        for param in sorted(list(all_parameters)):
                            row = {'Sección': pk.replace('_', ' ').title() if pk != 'main' else 'General', 'Parámetro': param}
                            
                            if pk == 'main':
                                iz_values = results_dict.get('iz', {}).get('valores', {})
                            else:
                                iz_values = results_dict.get('iz', {}).get(pk, {}).get('valores', {})

                            row['Izquierda (Valor)'] = round(iz_values.get(param, np.nan), 3) if isinstance(iz_values.get(param), (int, float)) else iz_values.get(param, np.nan)

                            if pk == 'main':
                                der_values = results_dict.get('der', {}).get('valores', {})
                                der_unit = results_dict.get('der', {}).get('unidades', {}).get(param, '-')
                            else:
                                der_values = results_dict.get('der', {}).get(pk, {}).get('valores', {})
                                der_unit = results_dict.get('der', {}).get(pk, {}).get('unidades', {}).get(param, '-')

                            row['Derecha (Valor)'] = round(der_values.get(param, np.nan), 3) if isinstance(der_values.get(param), (int, float)) else der_values.get(param, np.nan)
                            row['Unidad'] = der_unit

                            if not (pd.isna(row['Izquierda (Valor)']) and pd.isna(row['Derecha (Valor)'])):
                                combined_data.append(row)

                    if combined_data:
                        df_combined = pd.DataFrame(combined_data)
                        df_combined = df_combined[['Sección', 'Parámetro', 'Izquierda (Valor)', 'Derecha (Valor)', 'Unidad']]
                        df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        print(f"No hay datos detallados para exportar para {sheet_title}.")


            # Export all relevant sections using the generalized helper
            export_combined_data_to_excel(self.propiedades_vigas_sec, 'Prop Vigas Sec', writer)
            export_combined_data_to_excel(self.propiedades_vigas_trab, 'Prop Vigas Trabe', writer)
            export_combined_data_to_excel(self.data_vigas_sec, 'Resultados Vigas Sec', writer)
            export_combined_data_to_excel(self.data_vigas_trab, 'Resultados Vigas Trabe', writer)
            export_combined_data_to_excel(self.data_columna, 'Diseño Columna', writer)
            export_combined_data_to_excel(self.data_conexiones, 'Conexiones Secundarias', writer)


            # Advanced Connections (using the specialized export helper)
            def export_advanced_connection_results(results_dict, connection_name, writer):
                if not results_dict:
                    return

                sheet_name = f'Conexiones {connection_name}'[:31]
                
                all_parameters = set()
                for lado_data in results_dict.values():
                    for sub_section_data in lado_data.values():
                        if 'valores' in sub_section_data:
                            all_parameters.update(sub_section_data['valores'].keys())

                combined_data = []
                for sub_section in sorted({k for v in results_dict.values() for k in v.keys()}):
                    for param in sorted(list(all_parameters)):
                        row = {'Sección': sub_section.replace('_', ' ').title(), 'Parámetro': param}
                        
                        iz_values = results_dict.get('iz', {}).get(sub_section, {}).get('valores', {})
                        row['Izquierda (Valor)'] = round(iz_values.get(param, np.nan), 3) if isinstance(iz_values.get(param), (int, float)) else iz_values.get(param, np.nan)

                        der_values = results_dict.get('der', {}).get(sub_section, {}).get('valores', {})
                        der_unit = results_dict.get('der', {}).get(sub_section, {}).get('unidades', {}).get(param, '-')
                        row['Derecha (Valor)'] = round(der_values.get(param, np.nan), 3) if isinstance(der_values.get(param), (int, float)) else der_values.get(param, np.nan)
                        row['Unidad'] = der_unit

                        if not (pd.isna(row['Izquierda (Valor)']) and pd.isna(row['Derecha (Valor)'])):
                            combined_data.append(row)

                if combined_data:
                    df_combined = pd.DataFrame(combined_data)
                    df_combined = df_combined[['Sección', 'Parámetro', 'Izquierda (Valor)', 'Derecha (Valor)', 'Unidad']]
                    df_combined.to_excel(writer, sheet_name=sheet_name, index=False)

            export_advanced_connection_results(self.resultados_RBS, 'RBS', writer)
            export_advanced_connection_results(self.resultados_end_plate_4E, 'End Plate 4E', writer)
            export_advanced_connection_results(self.resultados_end_plate_4ES, 'End Plate 4ES', writer)
            export_advanced_connection_results(self.resultados_Diafragma_externo, 'Diafragma Externo', writer)
            export_advanced_connection_results(self.resultados_WUF_W, 'WUF-W', writer)

        print(f"¡Exportación a '{nombre_archivo}' completada exitosamente!")