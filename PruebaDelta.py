import numpy as np
import pandas as pd
from StructuralDesign import StructuralDesign

# Datos de entrada
materiales = {
    'A_sec': 36,
    'A_trab': 50,
    'E': 2100000,
    'fc': 210
}

datos_arquitectonicos = {
    'L1': 6.5,
    'L2': 6,
    'L3': 6.3,
    'L4': 6.5,
    'Pisos': 2,
    'Cm': 0.5,
    'Cv': 0.2,
    'He': 3,
    'Cu': 1.2 * 0.5 + 1.6 * 0.2
}

propiedades_vigas = {
    'vigas_sec_prop': {
        'iz': {'Acero': 36, 'bf': 12, 'tf': 0.6, 'h': 30, 'tw': 0.3},
        'der': {'Acero': 36, 'bf': 12, 'tf': 0.6, 'h': 31, 'tw': 0.4}
    },
    'vigas_trab_prop': {
        'iz': {'Acero': 50, 'bf': 16, 'tf': 1, 'h': 41, 'tw': 0.6},
        'der': {'Acero': 50, 'bf': 16, 'tf': 1, 'h': 37, 'tw': 0.5}
    },
    'vigas_secundarias': {
        'iz': {'Acero': 36, 'Lv': 6.3, 'Lt': 6.5, 'Vigas': 4},
        'der': {'Acero': 36, 'Lv': 6.3, 'Lt': 6, 'Vigas': 3}
    },
    'vigas_trab': {
        'iz': {'Acero': 50, 'Tipo': 'Central', 'Lv': 6.3, 'Lt': 6.5, 'Num': 4},
        'der': {'Acero': 50, 'Tipo': 'Central', 'Lv': 6.3, 'Lt': 6, 'Num': 3}
    }
}

propiedades_columna = {
    'columna': {'Acero': 50, 'Relleno hormigon': 'SI', 'b': 31, 'h': 31, 'e': 0.4},
    'Atiesadores': {
        'Sentido_b': {'Num': 1, 'I': 4, 'e': 0.3},
        'Sentido_h': {'Num': 1, 'I': 4, 'e': 0.3}
    }
}

propiedades_conexiones = {
    'Prop_RBS': {
        'iz': {'a': 8, 'b': 28, 'c': 3.6, 'E_pat': 80, 'tp': 1, 'hh': 1.2, 'bp': 5, 'E_alma': 60},
        'der': {'a': 8, 'b': 28, 'c': 3.6, 'E_pat': 80, 'tp': 1, 'hh': 1.2, 'bp': 5, 'E_alma': 60}
    },
    'Prop_end_plate_4E': {
        'iz': {'Acero': 50, 'Tipo': 'A490', 'Perno': 1, 'tp': 2.5, 'bp': 20, 'g': 15.2, 'pf': 4, 'E': 80},
        'der': {'Acero': 50, 'Tipo': 'A490', 'Perno': 1, 'tp': 2.5, 'bp': 20, 'g': 15.2, 'pf': 4, 'E': 80}
    },
    'Prop_end_plate_4ES': {
        'iz': {'Acero_ep': 50, 'Tipo': 'A490', 'Perno': 1, 'tp': 2.5, 'bp': 20, 'g': 15.2, 'pf': 4.4, 'de': 4.4, 'E': 80}, 
        'der': {'Acero_ep': 50, 'Tipo': 'A490', 'Perno': 1, 'tp': 2.5, 'bp': 20, 'g': 15.2, 'pf': 4, 'de': 4.4, 'E': 80}
    },
    'Diafragma_externo': {
        'iz': {'Ln': 30, 'td': 1.2, 'Acero_de': 50, 'E_patin_diaf': 60},
        'der': {'Ln': 32, 'td': 1.2, 'Acero_de': 50, 'E_patin_diaf': 60}
    },
    'Prop_WUF_W': {
        'iz': {'E': 100, 'tp': 0.6, 'a': 0.6, 'Acero': 36},
        'der': {'E': 100, 'tp': 0.6, 'a': 0.6, 'Acero': 36}
    }
}


parametros_conexiones = {
    'iz': {
        'angulo_apernado': {'Perno': 3/8, 'hp': 15, 'Lbord': 3, 'bp': 6},
        'angulo_soldado': {'E': 60, 'g': 0.3, 'hp': 15, 'tp': 0.3, 'bp': 6},
        'asiento_apernado': {'Perno': 3/8, 'bp': 6},
        'asiento_soldado': {'E': 60, 'bp': 5, 'tp': 0.3},
        'alma_alma': {'E': 60, 'hs': 15, 'g': 0.3}
    },
    'der': {
        'angulo_apernado': {'Perno': 3/8, 'hp': 15, 'Lbord': 3, 'bp': 6},
        'angulo_soldado': {'E': 60, 'g': 0.3, 'hp': 15, 'tp': 0.3, 'bp': 6},
        'asiento_apernado': {'Perno': 1/2, 'bp': 6},
        'asiento_soldado': {'E': 60, 'bp': 5, 'tp': 0.4},
        'alma_alma': {'E': 60, 'hs': 16, 'g': 0.3}
    }
}

# 1. Initialize the StructuralDesign object
structural_design = StructuralDesign(
    materiales,
    datos_arquitectonicos,
    propiedades_vigas,
    propiedades_columna,
    propiedades_conexiones,
    parametros_conexiones
)

structural_design._calcular_propiedades_iniciales()

# To calculate RBS connection for the 'iz' (left) side:
structural_design.calculate_rbs_connection('iz')

# # To calculate End Plate 4E connection for the 'der' (right) side:
# structural_design.calculate_end_plate_4e_connection('der')

# To calculate all advanced connections for both sides:
for lado in ['iz', 'der']:
    structural_design.calculate_rbs_connection(lado)
    # structural_design.calculate_end_plate_4e_connection(lado)
    # structural_design.calculate_end_plate_4es_connection(lado)
    # structural_design.calculate_Diafragma_externo_connection(lado)
    # structural_design.calculate_wuf_w_connection(lado)

# 3. After calculating, you can display or export the results:
structural_design.mostrar_resultados()
# structural_design.exportar_a_excel()
