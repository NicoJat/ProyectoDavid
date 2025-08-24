import openseespy.opensees as ops
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Rectangle

# Propiedades geométricas de la viga
L_beam = 6.0    # Longitud total de la viga (m)
h = 0.5         # Altura total de la sección (m)
tw = 0.01       # Espesor del alma (m)
tf = 0.02       # Espesor del ala (m)
A = 0.01        # Área de la sección (m²)
Ix = 8e-6       # Momento de inercia (m⁴)

# Propiedades del acero
Fy = 250e6      # Límite elástico (Pa)
E = 200e9       # Módulo de elasticidad (Pa)

def create_animation(node_coords_history, disp_list, moment_list, time_list, L_beam, a, c, Mp_full_section):
    """
    Crea una animación de la deformación de la viga y gráficos de histéresis
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Configurar subplot para la deformación de la viga
    ax1.set_xlim(-0.5, L_beam + 0.5)
    ax1.set_ylim(-max(abs(d) for d in disp_list) * 1.2, max(abs(d) for d in disp_list) * 1.2)
    ax1.set_xlabel('Posición (m)')
    ax1.set_ylabel('Desplazamiento vertical (m)')
    ax1.set_title('Deformación de la Viga RBS')
    ax1.grid(True, alpha=0.3)
    
    # Marcar la zona RBS
    rbs_start = a
    rbs_end = a + 2*c
    ax1.axvspan(rbs_start, rbs_end, alpha=0.2, color='red', label='Zona RBS')
    ax1.legend()
    
    # Configurar subplot para histéresis momento-rotación
    max_moment = max(abs(m) for m in moment_list)
    max_rotation = max(abs(d) for d in disp_list) / L_beam
    
    # Plot 0.8 Mp limits
    # ax2.axhline(y=0.8 * Mp_full_section, color='purple', linestyle='--', label='0.8 Mp')
    # ax2.axhline(y=-0.8 * Mp_full_section, color='purple', linestyle='--', label='-0.8 Mp')
    # ax2.legend(loc='best') # Asegurar que la leyenda muestre las nuevas etiquetas
    ax2.set_xlim(-max_rotation * 1.1, max_rotation * 1.1)
    ax2.set_ylim(-max_moment * 1.1, max_moment * 1.1)
    ax2.set_xlabel('Rotación (rad)')
    ax2.set_ylabel('Momento (N·m)')
    ax2.set_title('Curva de Histéresis Momento-Rotación')
    ax2.grid(True, alpha=0.3)
    
    # Configurar subplot para desplazamiento vs tiempo
    ax3.set_xlim(0, max(time_list))
    ax3.set_ylim(min(disp_list) * 1.1, max(disp_list) * 1.1)
    ax3.set_xlabel('Paso de análisis')
    ax3.set_ylabel('Desplazamiento (m)')
    ax3.set_title('Historia de Desplazamiento')
    ax3.grid(True, alpha=0.3)
    
    # Configurar subplot para momento vs tiempo
    ax4.set_xlim(0, max(time_list))
    ax4.set_ylim(min(moment_list) * 1.1, max(moment_list) * 1.1)
    ax4.set_xlabel('Paso de análisis')
    ax4.set_ylabel('Momento (N·m)')
    ax4.set_title('Historia de Momento')
    ax4.grid(True, alpha=0.3)
    
    # Inicializar líneas para la animación
    line_beam, = ax1.plot([], [], 'b-', linewidth=3, label='Viga deformada')
    points_nodes, = ax1.plot([], [], 'ro', markersize=8)
    line_hysteresis, = ax2.plot([], [], 'b-', linewidth=1.5)
    line_disp_time, = ax3.plot([], [], 'g-', linewidth=1.5)
    line_moment_time, = ax4.plot([], [], 'r-', linewidth=1.5)
    
    # Líneas de referencia (viga sin deformar)
    node_coords_initial = node_coords_history[0]
    x_initial = [node_coords_initial[i][0] for i in sorted(node_coords_initial.keys())]
    y_initial = [0] * len(x_initial)
    ax1.plot(x_initial, y_initial, 'k--', alpha=0.5, linewidth=1, label='Posición inicial')
    
    def animate(frame):
        if frame >= len(node_coords_history):
            return line_beam, points_nodes, line_hysteresis, line_disp_time, line_moment_time
        
        # Captura un segundo antes del final
        if frame == len(node_coords_history) - int(1000/50):  # 1000ms / 50ms interval = 20 frames antes
            plt.savefig('captura_animacion.png', dpi=300, bbox_inches='tight')
            print("Captura guardada: captura_animacion.png")
        
        # Actualizar deformación de la viga
        node_coords = node_coords_history[frame]
        x_data = [node_coords[i][0] for i in sorted(node_coords.keys())]
        y_data = [node_coords[i][1] for i in sorted(node_coords.keys())]
        
        line_beam.set_data(x_data, y_data)
        points_nodes.set_data(x_data, y_data)
        
        # Actualizar curva de histéresis
        if frame > 0:
            rotations = [d / L_beam for d in disp_list[:frame+1]]
            moments = moment_list[:frame+1]
            line_hysteresis.set_data(rotations, moments)
        
        # Actualizar historia de desplazamiento
        if frame > 0:
            times = time_list[:frame+1]
            disps = disp_list[:frame+1]
            line_disp_time.set_data(times, disps)
        
        # Actualizar historia de momento
        if frame > 0:
            times = time_list[:frame+1]
            moments = moment_list[:frame+1]
            line_moment_time.set_data(times, moments)
        
        return line_beam, points_nodes, line_hysteresis, line_disp_time, line_moment_time
    
    # Crear la animación
    anim = animation.FuncAnimation(fig, animate, frames=len(node_coords_history), 
                                 interval=17, blit=False, repeat=True)
    
    plt.tight_layout()
    return anim

def simular_viga_rbs_animacion(a, b, c, L_beam, animate=True):
    """
    Simula una viga con conexión RBS bajo carga cíclica
    """
    # Borrar modelo anterior
    ops.wipe()

    # Geometría reducida del ala (RBS)
    bf_total = 0.2
    bf_rbs = bf_total - 2 * b

    print(f"Configuración RBS:")
    print(f"  - Ancho total del ala: {bf_total:.3f} m")
    print(f"  - Ancho reducido del ala: {bf_rbs:.3f} m")
    print(f"  - Reducción por lado: {b:.3f} m")

    # Crear nodos
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0, 0)              # Apoyo empotrado
    ops.node(2, a, 0)              # Inicio de RBS
    ops.node(3, a + 2*c, 0)        # Final de RBS
    ops.node(4, L_beam, 0)         # Extremo libre (punto de carga)
    
    # Condiciones de apoyo (empotrado en nodo 1)
    ops.fix(1, 1, 1, 1)

    # Material acero no lineal (Steel02)
    ops.uniaxialMaterial("Steel02", 1, Fy, E, 0.03, 15, 0.925, 0.15)

    # Sección con fibras para la zona RBS
    ops.section("Fiber", 1)
    d_rbs = h - 2 * tf  # Altura del alma
    partes_espesor = 4
    partes_longitudinal = 10
    
    # Alma
    ops.patch("rect", 1, partes_espesor, partes_longitudinal, 
              -tw/2, -d_rbs/2, tw/2, d_rbs/2)
    # Ala superior
    ops.patch("rect", 1, partes_longitudinal, partes_espesor, 
              -bf_rbs/2, d_rbs/2, bf_rbs/2, d_rbs/2 + tf)
    # Ala inferior
    ops.patch("rect", 1, partes_longitudinal, partes_espesor, 
              -bf_rbs/2, -d_rbs/2 - tf, bf_rbs/2, -d_rbs/2)

    # Sección normal (fuera de RBS)
    ops.section("Fiber", 2)
    # Alma
    ops.patch("rect", 1, partes_espesor, partes_longitudinal, 
              -tw/2, -d_rbs/2, tw/2, d_rbs/2)
    # Ala superior
    ops.patch("rect", 1, partes_longitudinal, partes_espesor, 
              -bf_total/2, d_rbs/2, bf_total/2, d_rbs/2 + tf)
    # Ala inferior
    ops.patch("rect", 1, partes_longitudinal, partes_espesor, 
              -bf_total/2, -d_rbs/2 - tf, bf_total/2, -d_rbs/2)

    # Transformación geométrica (considera efectos P-Delta)
    ops.geomTransf("PDelta", 1)
    
    # Integración de Gauss-Lobatto
    ops.beamIntegration("Lobatto", 1, 1, 5)  # Para sección RBS
    ops.beamIntegration("Lobatto", 2, 2, 5)  # Para sección normal

    # Elementos de viga
    ops.element("forceBeamColumn", 1, 1, 2, 1, 2)  # Tramo inicial (sección normal)
    ops.element("forceBeamColumn", 2, 2, 3, 1, 1)  # Zona RBS (sección reducida)
    ops.element("forceBeamColumn", 3, 3, 4, 1, 2)  # Tramo final (sección normal)

    # Carga inicial pequeña para estabilizar el modelo
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(4, 0, -1e-6, 0)  # Pequeña carga vertical hacia abajo

    # Análisis de gravedad inicial
    ops.system("BandGeneral")
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.test("NormDispIncr", 1e-8, 50)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    
    ok = ops.analyze(1)
    if ok != 0:
        print("Error en análisis de gravedad inicial")
        return None, None, None, None, None

    # Niveles de deriva (rotación) y ciclos por nivel según AISC 341
    drift_levels = [0.00375, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04]
    cycles_per_level = [6, 6, 6, 4, 2, 2, 2, 2]
    steps_per_half_cycle = 10  # Pasos de análisis para cada medio ciclo

    # Listas para almacenar resultados
    disp_list = [ops.nodeDisp(4, 2)]
    moment_list = [ops.eleForce(2, 2)]  # Momento en el elemento RBS (elemento 2)
    time_list = [0]
    node_coords_history = [{i: (ops.nodeCoord(i, 1) + ops.nodeDisp(i, 1), 
                                ops.nodeCoord(i, 2) + ops.nodeDisp(i, 2)) 
                          for i in [1, 2, 3, 4]}]
    step_count = 1

    # Mantener las cargas de gravedad constantes durante el análisis cíclico
    ops.loadConst("-time", 0.0)

    print(f"\nIniciando análisis cíclico...")
    print(f"Total de niveles de deriva: {len(drift_levels)}")

    # Bucle principal de carga cíclica
    for i, drift in enumerate(drift_levels):
        num_cycles = cycles_per_level[i]
        target_disp_amplitude = drift * L_beam  # Amplitud de desplazamiento
        
        print(f"\nNivel {i+1}/{len(drift_levels)} - Deriva: {drift*100:.3f}% "
              f"(±{target_disp_amplitude*1000:.2f} mm) - {num_cycles} ciclos")

        for cycle in range(num_cycles):
            print(f"  Ciclo {cycle+1}/{num_cycles}")
            
            # Cada ciclo: Negativo -> Positivo
            for sign_idx, sign in enumerate([-1, 1]):
                target_disp = target_disp_amplitude * sign
                current_disp = ops.nodeDisp(4, 2)
                disp_to_travel = target_disp - current_disp
                increment = disp_to_travel / steps_per_half_cycle
                
                # Configurar integrador de control de desplazamiento
                ops.integrator("DisplacementControl", 4, 2, increment)
                ops.analysis("Static")
                
                # Realizar análisis paso a paso
                for step in range(steps_per_half_cycle):
                    ok = ops.analyze(1)
                    if ok != 0:
                        print(f"    Error en análisis - Deriva {drift*100:.3f}%, "
                              f"Ciclo {cycle+1}, {'Pos' if sign > 0 else 'Neg'}")
                        # Intentar con algoritmo modificado
                        ops.algorithm("ModifiedNewton", "-initial")
                        ok = ops.analyze(1)
                        if ok != 0:
                            ops.algorithm("Newton")
                            break
                        else:
                            ops.algorithm("Newton")
                    
                    # Guardar resultados
                    disp_list.append(ops.nodeDisp(4, 2))
                    moment_list.append(ops.eleForce(2, 2))  # Momento en zona RBS
                    time_list.append(step_count)
                    node_coords_history.append({
                        j: (ops.nodeCoord(j, 1) + ops.nodeDisp(j, 1), 
                            ops.nodeCoord(j, 2) + ops.nodeDisp(j, 2)) 
                        for j in [1, 2, 3, 4]
                    })
                    step_count += 1
                
                if ok != 0:
                    break
            if ok != 0:
                break
        if ok != 0:
            break

    # Crear animación si se solicita
    anim = None
    if animate and len(node_coords_history) > 1:
        print(f"\nCreando animación con {len(node_coords_history)} frames...")
        anim = create_animation(node_coords_history, disp_list, moment_list, 
                              time_list, L_beam, a, c)

    # Calcular valores máximos
    max_disp_abs = max(abs(d) for d in disp_list) if disp_list else 0
    max_moment_abs = max(abs(m) for m in moment_list) if moment_list else 0
    max_rotation = max_disp_abs / L_beam

    return max_rotation, max_moment_abs, disp_list, moment_list, anim

