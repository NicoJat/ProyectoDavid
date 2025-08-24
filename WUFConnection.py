class WUFConnection:
    def __init__(self, prop_wuf_w, prop_viga_trab, data_vigas_trab, columna, datos_arq):
        self.prop_wuf_w = prop_wuf_w
        self.prop_viga_trab = prop_viga_trab
        self.data_vigas_trab = data_vigas_trab
        self.columna = columna
        self.datos_arq = datos_arq

    def calcular(self):
        E = self.prop_wuf_w['E']
        tp = self.prop_wuf_w['tp']
        a = self.prop_wuf_w['a']
        Acero = self.prop_wuf_w['Acero']
        
        b_col = self.columna['b']
        h_col = self.columna['h']
        
        h = self.prop_viga_trab['valores']['h']
        bf = self.prop_viga_trab['valores']['bf']
        tf = self.prop_viga_trab['valores']['tf']
        Lt = self.datos_arq['L1']
        Ry = self.prop_viga_trab['valores']['Ry']
        fy = self.data_vigas_trab['valores']['fy']
        Zx = self.prop_viga_trab['valores']['Zx']
        
        Mpr = 1.2*Ry*fy*Zx/100000
        Vh = Mpr/(Lt-b_col/100)
        
        Ft_patin = Mpr*100/(h_col-tf)
        Rs_patin = bf*tf*E/14.23
        
        Vh_alma = Vh
        hh = 1.2
        hp = h-2*tf-2*hh-2*a
        Rs_alma_yield = hp*tp*0.6*Ry*Acero*1000/14.23/1000
        Rsol_alma = hp*tp*E/14.23
        
        Vu_placa_alma = Vh
        Rs_placa_alma = 0.32*E/14.23*hp*(tp-0.2)
        
        Resultados_WUF_W = {
            'Propiedades WUF-W': {
                'valores': {
                    'E': E, 'tp': tp, 'a': a, 'Acero': Acero
                },
                'unidades': {
                    'E': 'kgf/cm²', 'tp': 'cm', 'a': 'cm', 'Acero': 'kgf/cm²'
                }
            },
            
            'Calculos_conexion': { 
                'valores': {
                    'Mpr': Mpr, 'Vh': Vh
                },
                'unidades': {
                    'Mpr': 'tonf·m', 'Vh': 'tonf'
                }
            },
            
            'Chequeo patin': {
                'valores': {
                    'Ft': Ft_patin, 'Rs': Rs_patin
                },
                'unidades': {
                    'Ft': 'tonf', 'Rs': 'tonf'
                }
            },
            
            'Chequeo Alma': {
                'valores': {
                    'Vh_alma': Vh_alma, 'hh': hh, 'hp': hp,
                    'Rs_yield': Rs_alma_yield, 'Rsol': Rsol_alma
                },
                'unidades': {
                    'Vh_alma': 'tonf', 'hh': 'cm', 'hp': 'cm',
                    'Rs_yield': 'tonf', 'Rsol': 'tonf'
                }
            },
            
            'Placa-Alma': {
                'valores': {
                    'Vu': Vu_placa_alma, 'Rs': Rs_placa_alma 
                },
                'unidades': {
                    'Vu': 'tonf', 'Rs': 'tonf'
                }
            }
        }
        
        return Resultados_WUF_W