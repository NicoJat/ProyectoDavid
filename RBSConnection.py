import numpy as np

class RBSConnection:
    def __init__(self, prop_rbs, prop_viga_trab, data_vigas_trab, vigas_trabe, materiales, datos_arq):
        self.prop_rbs = prop_rbs
        self.prop_viga_trab = prop_viga_trab
        self.data_vigas_trab = data_vigas_trab
        self.vigas_trabe = vigas_trabe
        self.materiales = materiales
        self.datos_arq = datos_arq

    def calcular(self):
        a = self.prop_rbs['a']
        b = self.prop_rbs['b']
        c = self.prop_rbs['c']
        tp = self.prop_rbs['tp']
        hh = self.prop_rbs['hh']
        bp = self.prop_rbs['bp']
        E_pat = self.prop_rbs['E_pat']
        Ry = self.prop_viga_trab['valores']['Ry']
        fy = self.data_vigas_trab['valores']['fy']
        Zx = self.prop_viga_trab['valores']['Zx']
        Pi = self.data_vigas_trab['valores']['Pi']
        Num = self.vigas_trabe['Num']
        h = self.prop_viga_trab['valores']['h']
        tf = self.prop_viga_trab['valores']['tf']
        bf = self.prop_viga_trab['valores']['bf']
        tw = self.prop_viga_trab['valores']['tw']
        cv = self.data_vigas_trab['valores']['cv']
        E = self.materiales['E']
        
        sh = (a + b) / 2

        # Mpr de la viga
        Mpr_viga = (1.1 * Ry * Zx * fy + 500 * Pi * Num * sh) / 1e5

        # Reducción RBS
        Zrbs = Zx - 2 * c * tf * (h - tf)
        Mpr_con = 1.2 * Ry * fy * Zrbs / 1e5

        # Fuerzas de corte y nominal
        h_tw = h/tw
        Vu = 2 * Mpr_viga / (self.datos_arq['L1'] - sh/100) + Num * Pi / 2
        Vn = 0.6 * fy * (h - 2 * tf)*tw*cv / 1000

        Vu_conx = Vu

        # Momento final y momento plástico esperado
        Mf = Mpr_con + Vu_conx * sh / 100
        Mpe = Ry * fy * Zx / 1e5

        #Chequeo Patin
        Ft = 100 * Mf / (h - tf)
        Rs = bf*tf*E_pat/14.23
        
        #Chequeo Alma
        g = tp-0.2
        hp = h-2*tf-2*hh
        Rs_CJP = hp*tp*E/14.23
        Rs_filete = 0.32*E*g*bp*2*1.5/14.23+0.32*E*g*hp/14.23

        Resultados_RBS = {
            'Chequeo a corte': {
                'valores': {
                    'h/tw': h_tw,
                    'Vu': Vu,
                    'Vn': Vn
                },
                'unidades': {
                    'h/tw': '-',
                    'Vu': 'tonf',
                    'Vn': 'tonf'
                }
            },
            'Propiedades RBS': {
                'valores': {
                    'a': a,
                    'b': b,
                    'c': c,
                    'E_pat': E_pat
                },
                'unidades': {
                    'a': 'cm',
                    'b': 'cm',
                    'c': 'cm',
                    'E_pat': 'kgf/cm²'
                }
            },
            'Resultados_RBS': {
                'valores': {
                    'sh': sh,
                    'Mpr_viga': Mpr_viga,
                    'Zrbs': Zrbs,
                    'Mpr_con': Mpr_con,
                    'Vu': Vu,
                    'Vn': Vn,
                    'Mf': Mf,
                    'Mpe': Mpe
                },
                'unidades': {
                    'sh': 'cm',
                    'Mpr_viga': 'tonf·m',
                    'Zrbs': 'cm³',
                    'Mpr_con': 'tonf·m',
                    'Vu': 'tonf',
                    'Vn': 'tonf',
                    'Mf': 'tonf·m',
                    'Mpe': 'tonf·m'
                }
            },
            'Chequeo Patin': {
                'valores': {
                    'Ft': Ft,
                    'Rs': Rs
                },
                'unidades': {
                    'Ft': 'tonf',
                    'Rs': 'tonf'
                }
            },
            'Chequeo Alma': {
                'valores': {
                    'g':g,
                    'hp':hp,
                    'Rs_CJP':Rs_CJP,
                    'Rs_filete':Rs_filete
                },
                'unidades': {
                    'g': 'cm',
                    'hp': 'cm',
                    'Rs_CJP': 'tonf',
                    'Rs_filete': 'tonf'
                }
            }
        }
        return Resultados_RBS