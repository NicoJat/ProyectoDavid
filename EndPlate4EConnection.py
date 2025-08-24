import numpy as np

class EndPlate4EConnection:
    def __init__(self, prop_end_plate_4e, prop_viga_trab, data_vigas_trab, columna, datos_arq):
        self.prop_end_plate_4e = prop_end_plate_4e
        self.prop_viga_trab = prop_viga_trab
        self.data_vigas_trab = data_vigas_trab
        self.columna = columna
        self.datos_arq = datos_arq

    def calcular(self):
        Ry = self.prop_viga_trab['valores']['Ry']
        fy = self.data_vigas_trab['valores']['fy']
        Zx = self.prop_viga_trab['valores']['Zx']
        h = self.prop_viga_trab['valores']['h']
        bf = self.prop_viga_trab['valores']['bf']
        Lt = self.datos_arq['L1']
        tf = self.prop_viga_trab['valores']['tf']
        tw = self.prop_viga_trab['valores']['tw']
        A = self.prop_viga_trab['valores']['A']
        b_col = self.columna['b']
        e_col = self.columna['e']
        
        g = self.prop_end_plate_4e['g']
        bp = self.prop_end_plate_4e['bp']
        pf = self.prop_end_plate_4e['pf']
        Perno = self.prop_end_plate_4e['Perno']
        Tipo = self.prop_end_plate_4e['Tipo']
        Acero = self.prop_end_plate_4e['Acero']
        E = self.prop_end_plate_4e['E']
        
        Mpr = 1.2*Ry*fy*Zx/100000
        Sh = min(h/2,3*bf)
        Vu = 2*Mpr/(Lt-b_col/100-2*Sh/100)+b_col*e_col/2
        Mf = Mpr+Vu*Sh/100
        
        Esfv = 0.35*Vu/Perno**2
        if Tipo =="A307":
            Efst = min(59-1.9*Esfv,45) 
        elif Tipo =="A325":
            Efst = min(117-1.9*Esfv,90) 
        else:
            Efst = min(147-1.9*Esfv,113) 
            
        Estreal = 70.18*Mf/(h-tf)/(Perno**2)
        
        h0 = h-tf/2+pf
        h1 = h-tf/2-tf-pf
        s = (bp*g)**0.5/2
        pfc = s if pf>s else pf 
        Yp = bp/2*(h1*(1/pfc+1/s)+h0/pfc-0.5)+2/g*(h1*(pfc+s))
        
        tp_min = 39.72*(Mf/Acero/Yp)**0.5
        
        # Chequeos
        Ft = Mf*100/(h-tf)
        Rs_patin = bf*tf*E/14.23+0.32*E*0.8*(bf-tw-2)/14.23*1.5
        
        Vu_alma = Vu
        Rs_alma = (h-2*tf)*tw*E/14.23
        
        Zp = min(h, 3*bf)
        g_refuerzo = max(0.75*A,0.6)
        
        End_Plate_4E = {
            'Propiedades_end_plate': {
                'valores': {
                    'tf': tf, 'bf': bf, 'h': h, 'g': g, 'bp': bp, 'pf': pf,
                    'Perno': Perno, 'Tipo': Tipo, 'Acero': Acero, 'E': E
                },
                'unidades': {
                    'tf': 'cm', 'bf': 'cm', 'h': 'cm', 'g': 'cm', 'bp': 'cm', 'pf': 'cm',
                    'Perno': 'cm', 'Tipo': '-', 'Acero': 'kgf/cm²', 'E': 'kgf/cm²'
                }
            },
            
            'Resultados': {
                'valores': {
                    'Mpr': Mpr, 'Sh': Sh, 'Vu': Vu, 'Mf': Mf, 'Esfv': Esfv,
                    'Efst': Efst, 'Estreal': Estreal, 'tp_min': tp_min
                },
                'unidades': {
                    'Mpr': 'tonf·m', 'Sh': 'cm', 'Vu': 'tonf', 'Mf': 'tonf·m', 'Esfv': 'kgf/cm²',
                    'Efst': 'kgf/cm²', 'Estreal': 'kgf/cm²', 'tp_min': 'cm'
                }
            },
            
            'Chequeo Patin': {
                'valores': {
                    'Ft': Ft, 'E': E, 'Rs': Rs_patin
                },
                'unidades': {
                    'Ft': 'tonf', 'E': 'kgf/cm²', 'Rs': 'tonf'
                }
            },
            
            'Chequeo Alma': {
                'valores': {
                    'Vu': Vu_alma, 'Rs': Rs_alma
                },
                'unidades': {
                    'Vu': 'tonf', 'Rs': 'tonf'
                }
            },
            
            'Refuerzo Adicional': {
                'valores': {
                    'Zp': Zp, 'g': g_refuerzo
                },
                'unidades': {
                    'Zp': 'cm', 'g': 'cm'
                }
            }
        }
        
        return End_Plate_4E