class DiafragmaExterno:
    def __init__(self, prop_end_plate_4es, prop_viga_trab, data_vigas_trab, columna, data_columna, vigas_trabe, datos_arq):
        self.prop_end_plate_4es = prop_end_plate_4es
        self.prop_viga_trab = prop_viga_trab
        self.data_vigas_trab = data_vigas_trab
        self.columna = columna
        self.data_columna = data_columna
        self.vigas_trabe = vigas_trabe
        self.datos_arq = datos_arq

    def calcular(self):
        
        Ry = self.prop_viga_trab['valores']['Ry']
        fy = self.data_vigas_trab['valores']['fy']
        Zx = self.prop_viga_trab['valores']['Zx']
        h = self.prop_viga_trab['valores']['h']
        bf = self.prop_viga_trab['valores']['bf']
        Pi = self.data_vigas_trab['valores']['Pi']
        Num = self.vigas_trabe['Num']
        Lt = self.datos_arq['L1']
        
        tf = self.prop_viga_trab['valores']['tf']
        tw = self.prop_viga_trab['valores']['tw']
        
        Ln = self.prop_end_plate_4es['Ln']
        td = self.prop_end_plate_4es['td']
        
        b_col = self.columna['b']
        e_col = self.columna['e']
        h_col = self.columna['h']
        
        fy_rell_col = self.data_columna['columna']['valores']['fy']
        Acero_de = self.prop_end_plate_4es['Acero_de']
        
        E_pat_diaf = self.prop_end_plate_4es['E_patin_diaf']
        
        dj = h-2*td
        bj = b_col-2*e_col
        
        fyd = Acero_de*1000/14.23
        Zplbwn = tw*(h-2*tf)**2/4
        m = 4*e_col/dj*(bj*fy_rell_col/tw/fy)**0.5
        
        Mpl = 1.2*Ry*fy*Zx/100000
        Mcf = (Lt/5)/(Lt/5-Ln/100)*1.3*Mpl
        hd = (2*Ln+h_col-bf)/4-(h_col/2-bf/2)
        Pbf = (2.86*(4*e_col+td)*e_col*fy_rell_col+3.3*hd*td*fyd)/1000
        Mbfu = Pbf*(h-tf)/100
        Mbwn = m*Zplbwn*fy/100000
        Mjcf = Mbfu+Mbwn
        
        E_pat_diaf = self.prop_end_plate_4es['E_patin_diaf']
        
        Ft_pat_diaf = Pbf
        g_pat_diaf = min(td,tf)-0.2
        Rs_pat_diaf = 0.32*E_pat_diaf/14.23*g_pat_diaf*bf*1.5+0.32*E_pat_diaf/14.23*g_pat_diaf*Ln*2
        R_falt = Ft_pat_diaf-Rs_pat_diaf
        L_ext = R_falt*14.23/0.32/E_pat_diaf/g_pat_diaf/2
        Rs_pat_col = bf*tf*E_pat_diaf/14.23+b_col*td*E_pat_diaf/14.23
        
        Vu_alma_final = Mcf/Lt+Num*Pi/2
        Rs_alma_final = (h-2*tf)*tw*E_pat_diaf/14.23
        
        DiafragmaExterno = {
            
            'Diafragma_externo': {
                'valores': {
                    'Mpl': Mpl, 'Mcf_req': Mcf, 'hd': hd, 'Pbf': Pbf,
                    'Mbfu': Mbfu, 'Mbwn': Mbwn, 'Mjcf_cap': Mjcf
                },
                'unidades': {
                    'Mpl': 'tonf·m', 'Mcf_req': 'tonf·m', 'hd': 'cm', 'Pbf': 'tonf',
                    'Mbfu': 'tonf·m', 'Mbwn': 'tonf·m', 'Mjcf_cap': 'tonf·m'
                }
            },
            
            'Chequeo Patín-diafragma': {
                'valores': {
                    'Ft_pat_diaf': Ft_pat_diaf, 'g_pat_diaf': g_pat_diaf, 'Rs_pat_diaf': Rs_pat_diaf,
                    'R_falt': R_falt, 'L_ext': L_ext, 'Rs_pat_col': Rs_pat_col
                },
                'unidades': {
                    'Ft_pat_diaf': 'tonf', 'g_pat_diaf': 'cm', 'Rs_pat_diaf': 'tonf',
                    'R_falt': 'tonf', 'L_ext': 'cm', 'Rs_pat_col': 'tonf'
                }
            },
            
            'Chequeo Alma Final': {
                'valores': {
                    'Vu':Vu_alma_final, 'Rs_alma':Rs_alma_final
                },
                'unidades': {
                    'Vu':'tonf', 'Rs_alma':'tonf'
                }
            }
        }
        
        return DiafragmaExterno