import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, Toplevel
import math
import inspect
from datetime import datetime

# --- 常量定义 ---
# Existing constants from your file
R_GAS_CONSTANT = 8.314
SPEED_OF_LIGHT = 299792458
GRAVITATIONAL_ACCELERATION = 9.80665
AVOGADRO_CONSTANT = 6.022e23
PLANCK_CONSTANT = 6.626e-34
GRAVITATIONAL_CONSTANT_G = 6.67430e-11
ELEMENTARY_CHARGE = 1.602176634e-19
FARADAY_CONSTANT_DERIVED = ELEMENTARY_CHARGE * AVOGADRO_CONSTANT
K_COULOMB = 8.9875e9
V_SOUND_AIR_STANDARD = 343
KB_WATER_CRYOSCOPIC = 1.86
KE_WATER_EBULLIOSCOPIC = 0.512
STP_MOLAR_VOLUME = 22.414
PURE_WATER_FREEZING_POINT_C = 0.0
PURE_WATER_BOILING_POINT_C = 100.0
STANDARD_ATM_PRESSURE_Pa = 101325.0
MOLAR_MASS_O2_g_mol = 32.00
MOLAR_MASS_CO2_g_mol = 44.01

# NEW Constants added
BOLTZMANN_CONSTANT_k = 1.380649e-23 # J/K
WIEN_DISPLACEMENT_CONSTANT_b = 2.898e-3 # m·K
STEFAN_BOLTZMANN_CONSTANT_sigma = 5.670374e-8 # W/(m²·K⁴)
PERMITTIVITY_FREE_SPACE_epsilon0 = 8.854187817e-12 # F/m
PERMEABILITY_FREE_SPACE_mu0 = 4 * math.pi * 1e-7 # N/A² or H/m
RYDBERG_CONSTANT_R_inf = 10973731.56816 # m⁻¹ (for Bohr model)
GAS_CONSTANT_L_ATM = 0.082057 # L·atm/(mol·K)
GAS_CONSTANT_MPA_L = 0.008314 # MPa·L/(mol·K)

# --- Physical/Chemical Constants (for display in constants window) ---
CONSTANTS_REFERENCE_DATA = [
    {"name": "理想气体常数 (R)", "value": R_GAS_CONSTANT, "unit": "J/(mol·K)"}, # Using Chinese names for display
    {"name": "理想气体常数 (R)", "value": GAS_CONSTANT_L_ATM, "unit": "L·atm/(mol·K)"},
    {"name": "理想气体常数 (R)", "value": GAS_CONSTANT_MPA_L, "unit": "MPa·L/(mol·K)"},
    {"name": "光速 (c)", "value": SPEED_OF_LIGHT, "unit": "m/s"},
    {"name": "标准重力加速度 (g)", "value": GRAVITATIONAL_ACCELERATION, "unit": "m/s²"},
    {"name": "阿伏伽德罗常数 (Nᴀ)", "value": AVOGADRO_CONSTANT, "unit": "mol⁻¹"},
    {"name": "普朗克常数 (h)", "value": PLANCK_CONSTANT, "unit": "J·s"},
    {"name": "万有引力常数 (G)", "value": GRAVITATIONAL_CONSTANT_G, "unit": "N·m²/kg²"},
    {"name": "元电荷 (e)", "value": ELEMENTARY_CHARGE, "unit": "C"},
    {"name": "法拉第常数 (F)", "value": FARADAY_CONSTANT_DERIVED, "unit": "C/mol"},
    {"name": "库伦常数 (k)", "value": K_COULOMB, "unit": "N·m²/C²"},
    {"name": "标准声速 (空气20°C)", "value": V_SOUND_AIR_STANDARD, "unit": "m/s"},
    {"name": "STP气体摩尔体积 (Vm)", "value": STP_MOLAR_VOLUME, "unit": "L/mol (STP)"},
    {"name": "标准大气压 (1 atm)", "value": STANDARD_ATM_PRESSURE_Pa, "unit":"Pa"},
    {"name": "玻尔兹曼常数 (k)", "value": BOLTZMANN_CONSTANT_k, "unit": "J/K"},
    {"name": "维恩位移定律常数 (b)", "value": WIEN_DISPLACEMENT_CONSTANT_b, "unit": "m·K"},
    {"name": "斯特凡-玻尔兹曼常数 (σ)", "value": STEFAN_BOLTZMANN_CONSTANT_sigma, "unit": "W/(m²·K⁴)"},
    {"name": "真空电容率 (ε₀)", "value": PERMITTIVITY_FREE_SPACE_epsilon0, "unit": "F/m"},
    {"name": "真空磁导率 (μ₀)", "value": PERMEABILITY_FREE_SPACE_mu0, "unit": "H/m"},
    {"name": "里德伯常数 (R∞)", "value": RYDBERG_CONSTANT_R_inf, "unit": "m⁻¹"},

]


# === CALCULATION FUNCTIONS (Existing + ALL previous new ones + EVEN MORE NEW ones) ===
# Physics
def calc_kinematics_v_uat(u, a, t):
    if t < 0: return {"error": "时间 t 不能为负"}
    v = u + a * t
    return {"v": v}
def calc_kinematics_s_ut_half_at2(u, a, t):
    if t < 0: return {"error": "时间 t 不能为负"}
    s = u * t + 0.5 * a * t**2
    return {"s": s}
def calc_force_f_ma(m, a):
    if m < 0: return {"error": "质量 m 不能为负"}
    f = m * a
    return {"F": f}
def calc_energy_ke(m, v):
    if m < 0: return {"error": "质量 m 不能为负"}
    ke = 0.5 * m * v**2
    return {"KE": ke}
def calc_energy_pe_mgh(m, h):
    if m < 0: return {"error": "质量 m 不能为负"}
    pe = m * GRAVITATIONAL_ACCELERATION * h
    return {"PE": pe}
def calc_einstein_emc2(m):
    if m < 0: return {"error": "质量 m 不能为负"}
    e = m * SPEED_OF_LIGHT**2
    return {"E": e}
def calc_ohms_law_voltage(i, r):
    if r < 0: return {"error": "电阻 R 不能为负"}
    v = i * r
    return {"V": v}
def calc_ohms_law_current(v, r):
    if r == 0: return {"error": "电阻 R 不能为零"}
    if r < 0: return {"error": "电阻 R 不能为负"}
    i = v / r
    return {"I": i}
def calc_ohms_law_resistance(v, i):
    if i == 0: return {"error": "电流 I 不能为零"}
    r_calc = v / i
    if r_calc < 0: return {"error": "计算得到的电阻 R 为负 (检查V,I方向或输入)"}
    return {"R": r_calc}
def calc_power_iv(i, v):
    p = i * v
    return {"P_watt": p}
def calc_photon_energy_freq(f):
    if f < 0: return {"error": "频率 f 不能为负"}
    e = PLANCK_CONSTANT * f
    return {"E_photon": e}
def calc_photon_energy_wavelength(lambda_val):
    if lambda_val <= 0: return {"error": "波长 λ 必须为正"}
    e = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / lambda_val
    return {"E_photon": e}
def calc_universal_gravitation(m1, m2, r):
    if m1 < 0 or m2 < 0: return {"error": "质量 m1, m2 不能为负"}
    if r <= 0: return {"error": "距离 r 必须为正"}
    F_gravity = (GRAVITATIONAL_CONSTANT_G * m1 * m2) / (r**2)
    return {"F_gravity": F_gravity}
def calc_pendulum_period(L):
    if GRAVITATIONAL_ACCELERATION == 0: return {"error": "重力加速度g不能为零(常数设定问题)"}
    if L <= 0: return {"error": "摆长 L 必须为正"}
    T_pendulum = 2 * math.pi * math.sqrt(L / GRAVITATIONAL_ACCELERATION)
    return {"T_pendulum": T_pendulum}
def calc_projectile_motion(v0, angle_deg, h0=0):
    if v0 < 0: return {"error": "初速度v₀不能为负"}
    if h0 < 0: return {"error": "初始高度h₀不能为负"}
    angle_rad = math.radians(angle_deg)
    g = GRAVITATIONAL_ACCELERATION
    H_above_h0 = (v0**2 * (math.sin(angle_rad))**2) / (2 * g) if g != 0 else float('inf')
    max_height_abs = h0 + H_above_h0
    vy0 = v0 * math.sin(angle_rad)
    discriminant = vy0**2 + 2 * g * h0
    if discriminant < 0 and not (abs(h0)<1e-9 and abs(vy0)<1e-9) : return {"error": "炮弹无法达到y=0的水平面（或参数导致判别式为负）。"}
    if g == 0:
        if abs(math.cos(angle_rad)) < 1e-9: time_of_flight = float('inf') if vy0 > 0 else (0 if h0==0 else float('inf') if vy0==0 else -h0/vy0 if vy0 < 0 else float('inf') ) ; range_val = 0
        else: time_of_flight = float('inf') if h0>0 or vy0!=0 else 0 ; range_val = float('inf') if time_of_flight > 0 else 0
    else:
        t_flight_to_zero_plus = (vy0 + math.sqrt(max(0,discriminant))) / g
        t_flight_to_zero_minus = (vy0 - math.sqrt(max(0,discriminant))) / g
        time_of_flight = 0
        if t_flight_to_zero_plus > 1e-9 : time_of_flight = t_flight_to_zero_plus
        elif t_flight_to_zero_minus > 1e-9 : time_of_flight = t_flight_to_zero_minus
        elif abs(h0)<1e-9 and abs(vy0)<1e-9: time_of_flight = 0.0
        else: return {"error": "无法计算有效的正飞行时间以到达y=0。检查发射参数。"}
    if time_of_flight < 0 : time_of_flight = 0.0
    range_val = v0 * math.cos(angle_rad) * time_of_flight
    return {"range": range_val, "max_height_abs": max_height_abs, "time_of_flight": time_of_flight}
def calc_wave_speed(frequency, wavelength):
    if frequency < 0: return {"error": "频率不能为负"}
    v_wave = frequency * wavelength; return {"v_wave": v_wave}
def calc_specific_heat_energy(mass, specific_heat_capacity, delta_temp):
    if mass < 0 or specific_heat_capacity < 0 : return {"error": "质量和比热容不能为负"}
    Q_heat = mass * specific_heat_capacity * delta_temp; return {"Q_heat": Q_heat}
def calc_coulombs_law_force(q1, q2, r_distance):
    if r_distance <= 0: return {"error": "电荷间距离r必须为正"}
    F_electric = (K_COULOMB * abs(q1 * q2)) / (r_distance**2); return {"F_electric": F_electric}
def calc_capacitance(Q_charge, V_potential_diff):
    if abs(V_potential_diff) < 1e-9:
        if abs(Q_charge) < 1e-9: return {"C_capacitance": float('inf'), "info": "V和Q接近零, C未定义/无限."}
        else: return {"error": "电势差V为零时,若电荷Q非零,则电容无限大."}
    C_capacitance = Q_charge / V_potential_diff
    if C_capacitance < 0: return {"error":"计算的电容C为负, Q和V符号可能不适用于典型物理电容器."}
    return {"C_capacitance": C_capacitance}
def calc_capacitor_energy(C_capacitance, V_potential_diff):
    if C_capacitance < 0: return {"error": "电容C不能为负"}
    U_capacitor = 0.5 * C_capacitance * (V_potential_diff**2); return {"U_capacitor": U_capacitor}
def calc_resistors_series_two(R1, R2):
    if R1 < 0 or R2 < 0 : return {"error":"电阻值不能为负"}
    R_total_series = R1 + R2; return {"R_total_series": R_total_series}
def calc_resistors_parallel_two(R1, R2):
    if R1 < 0 or R2 < 0 : return {"error":"电阻值不能为负"}
    if abs(R1) < 1e-9 and abs(R2) < 1e-9: return {"R_total_parallel": 0}
    if abs(R1) < 1e-9 or abs(R2) < 1e-9: return {"R_total_parallel": 0}
    if abs(R1 + R2) < 1e-9: return {"error":"R1+R2接近零,导致除零或结果过大."}
    R_total_parallel = (R1 * R2) / (R1 + R2); return {"R_total_parallel": R_total_parallel}
def calc_snells_law_theta2(n1, theta1_deg, n2):
    if n1 <=0 or n2 <=0: return {"error": "折射率n1,n2必须为正"}
    if not (0 <= theta1_deg <= 90): return {"error": "入射角 θ₁ 必须在 0 到 90 度之间"}
    theta1_rad = math.radians(theta1_deg); sin_theta2_val = (n1 / n2) * math.sin(theta1_rad)
    if abs(sin_theta2_val - 1.0) < 1e-9 : sin_theta2_val = 1.0
    if abs(sin_theta2_val + 1.0) < 1e-9 : sin_theta2_val = -1.0
    if sin_theta2_val > 1.0:
        if n1 > n2: return {"theta2_info_content": "全内反射 (TIR)", "sin_theta2_calc_val": round(sin_theta2_val,6)}
        else: return {"error": f"sin(θ₂) = {sin_theta2_val:.4f} > 1, 但 n1 <= n2. 输入可能不合逻辑."}
    elif sin_theta2_val < -1.0: return {"error": f"计算值超出范围 (sin(θ₂) = {sin_theta2_val:.4f} < -1)"}
    else:
        theta2_rad = math.asin(sin_theta2_val); theta2_deg_val = math.degrees(theta2_rad)
        return {"theta2_deg_val": theta2_deg_val, "sin_theta2_calc_val": round(sin_theta2_val,6)}
def calc_lens_formula_di(f_focal_length, do_object_distance):
    if abs(f_focal_length) < 1e-9 : return {"error":"焦距f不能为零"}
    if abs(do_object_distance - f_focal_length) < 1e-9: di_image_distance = float('inf'); magnification_M = float('-inf') if f_focal_length > 0 else float('+inf')
    elif abs(do_object_distance) < 1e-9: di_image_distance = 0; magnification_M = 1.0
    else:
        numerator = f_focal_length * do_object_distance; denominator = do_object_distance - f_focal_length
        if abs(denominator) < 1e-9: di_image_distance = float('inf')
        else: di_image_distance = numerator / denominator
        if abs(do_object_distance) < 1e-9: magnification_M = float('nan')
        elif math.isinf(di_image_distance): magnification_M = float('-inf') if di_image_distance > 0 else float('+inf'); magnification_M *= -1 if do_object_distance < 0 else 1
        else: magnification_M = -di_image_distance / do_object_distance
    return {"di_image_distance": di_image_distance, "magnification_M": magnification_M}
def calc_de_broglie_wavelength(mass_kg, velocity_ms):
    if mass_kg <= 0 : return {"error":"质量m必须为正"}
    momentum_p = mass_kg * velocity_ms
    if abs(momentum_p) < 1e-18: return {"lambda_debroglie": float('inf'), "info":"动量 (mv) 接近零, 波长趋于无穷大."}
    lambda_debroglie = PLANCK_CONSTANT / momentum_p; return {"lambda_debroglie": lambda_debroglie}
def calc_power_i2r(current_A, resistance_ohm):
    if resistance_ohm < 0 : return {"error":"电阻R不能为负"}
    P_power = (current_A**2) * resistance_ohm; return {"P_power_watt": P_power}
def calc_power_v2r(voltage_V, resistance_ohm):
    if resistance_ohm < 0 : return {"error":"电阻R不能为负"}
    if abs(resistance_ohm) < 1e-9:
        if abs(voltage_V) < 1e-9: return {"P_power_watt": 0, "info":"V和R接近零, P视为0."}
        else: return {"P_power_watt": float('inf'), "info":"R接近零且V非零, P趋于无穷大."}
    P_power = (voltage_V**2) / resistance_ohm; return {"P_power_watt": P_power}
def calc_doppler_sound_source_away_observer_still(f_source, v_source_speed):
    if f_source < 0 or V_SOUND_AIR_STANDARD <=0: return {"error": "声源频率或声速不能为负/零。"}
    denominator = V_SOUND_AIR_STANDARD + v_source_speed
    if abs(denominator) < 1e-9 : return {"error": "分母(v_sound + v_src)为零."}
    f_observed = f_source * (V_SOUND_AIR_STANDARD / denominator); return {"f_observed_Hz": f_observed}
def calc_doppler_sound_source_towards_observer_still(f_source, v_source_speed):
    if f_source < 0 or V_SOUND_AIR_STANDARD <=0: return {"error": "声源频率或声速不能为负/零。"}
    denominator = V_SOUND_AIR_STANDARD - v_source_speed
    if abs(denominator) < 1e-9: return {"f_observed_Hz": float('inf'), "info": "源速等于声速 (激波)."}
    elif denominator < 0: return {"error":"源速超音速朝向观察者."}
    f_observed = f_source * (V_SOUND_AIR_STANDARD / denominator); return {"f_observed_Hz": f_observed}
def calc_average_velocity(distance, time):
    if time <= 0 or distance < 0: return {"error": "时间需为正, 路程不能为负。"}
    v_avg = distance / time; return {"v_avg": v_avg}
def calc_density(mass, volume):
    if mass < 0 or volume <= 0: return {"error": "质量不能为负, 体积需为正。"}
    rho = mass / volume; return {"density_rho": rho}
def calc_work_fd_cos_theta(force, distance, angle_deg=0):
    if force < 0 or distance < 0 : return {"error": "力F和位移d大小不能为负。"}
    angle_rad = math.radians(angle_deg); work = force * distance * math.cos(angle_rad)
    return {"work_W": work}
def calc_pressure_FA(force, area):
    if force < 0 or area <= 0: return {"error": "力F大小不能为负, 面积A需为正。"}
    pressure = force / area; return {"pressure_P": pressure}
def calc_hookes_law_force(k_spring_constant, x_displacement):
    if k_spring_constant < 0: return {"error": "弹簧常数k不能为负。"}
    force_spring = k_spring_constant * abs(x_displacement); return {"force_spring_F": force_spring}
def calc_hookes_law_displacement(F_force_on_spring, k_spring_constant):
    if F_force_on_spring < 0 or k_spring_constant <= 0: return {"error":"力F大小不能为负, 弹簧常数k需为正。"}
    x_displacement = F_force_on_spring / k_spring_constant; return {"x_displacement": x_displacement}
def calc_spring_potential_energy(k_spring_constant, x_displacement):
    if k_spring_constant < 0: return {"error": "弹簧常数k不能为负。"}
    PE_spring = 0.5 * k_spring_constant * (x_displacement**2); return {"PE_spring": PE_spring}
def calc_centripetal_force(mass, velocity_tangential, radius_circular_path):
    if mass < 0 or radius_circular_path <= 0: return {"error": "质量m不能为负, 轨道半径r需为正。"}
    Fc = (mass * velocity_tangential**2) / radius_circular_path; return {"Fc_centripetal_force": Fc}
def calc_escape_velocity(M_celestial_body_mass, R_celestial_body_radius):
    if M_celestial_body_mass <= 0 or R_celestial_body_radius <= 0: return {"error": "天体质量M和半径R必须为正"}
    val_inside_sqrt = (2 * GRAVITATIONAL_CONSTANT_G * M_celestial_body_mass) / R_celestial_body_radius
    if val_inside_sqrt <0: return {"error":"计算逃逸速度时根号内值为负."}
    v_esc = math.sqrt(val_inside_sqrt); return {"v_escape": v_esc}
def calc_photon_momentum_from_wavelength(lambda_wavelength):
    if lambda_wavelength <= 0: return {"error": "波长λ必须为正"}
    p_momentum = PLANCK_CONSTANT / lambda_wavelength; return {"p_photon_momentum": p_momentum}
def calc_photon_momentum_from_energy(E_photon_energy):
    if E_photon_energy < 0 : return {"error": "光子能量E不能为负"}
    if SPEED_OF_LIGHT == 0: return {"error": "光速c常数不能为零."}
    p_momentum = E_photon_energy / SPEED_OF_LIGHT; return {"p_photon_momentum": p_momentum}
def calc_angular_velocity_v_r(v_linear_velocity, r_radius):
    if r_radius <= 0: return {"error": "半径r必须为正"}
    omega = v_linear_velocity / r_radius; return {"omega_angular_velocity": omega}
def calc_buoyancy_force(fluid_density_kg_m3, volume_displaced_m3):
    if fluid_density_kg_m3 < 0 or volume_displaced_m3 < 0: return {"error":"密度和体积不能为负"}
    F_buoyancy = fluid_density_kg_m3 * volume_displaced_m3 * GRAVITATIONAL_ACCELERATION; return {"F_buoyancy_N": F_buoyancy}
def calc_pressure_depth_rho_g_h(fluid_density_kg_m3, depth_h_m):
    if fluid_density_kg_m3 < 0 or depth_h_m < 0: return {"error":"密度和深度不能为负"}
    P_fluid = fluid_density_kg_m3 * GRAVITATIONAL_ACCELERATION * depth_h_m; return {"P_fluid_Pa": P_fluid}
def calc_work_ideal_gas_isobaric(pressure_Pa, V_initial_m3, V_final_m3):
    if pressure_Pa < 0 : return {"error":"压强不能为负"}
    delta_V_m3 = V_final_m3 - V_initial_m3; W_by_gas_J = pressure_Pa * delta_V_m3
    return {"W_by_gas_J": W_by_gas_J, "delta_V_m3": delta_V_m3}
def calc_thermal_efficiency_carnot(T_hot_K, T_cold_K):
    if T_hot_K <= 0 or T_cold_K <= 0 or T_hot_K <= T_cold_K: return {"error":"绝对温度需为正且Th > Tc"}
    efficiency_carnot = 1.0 - (T_cold_K / T_hot_K); return {"efficiency_carnot_percent": efficiency_carnot * 100}
def calc_magnetic_force_on_charge_qvBsin(q_charge_C, v_velocity_ms, B_field_T, angle_deg_v_B=90):
    if B_field_T <0 : return {"error":"磁场强度B不能为负"}
    angle_rad = math.radians(angle_deg_v_B); F_magnetic = abs(q_charge_C) * v_velocity_ms * B_field_T * abs(math.sin(angle_rad))
    return {"F_magnetic_N": F_magnetic}
def calc_time_constant_RL_circuit(L_inductance_H, R_resistance_ohm):
    if L_inductance_H <0 or R_resistance_ohm <0 : return {"error": "电感L和电阻R不能为负"}
    if R_resistance_ohm == 0: return {"error":"纯电感电路(R=0), 时间常数未定义/无穷"} if L_inductance_H !=0 else {"tau_RL_s":0.0}
    tau_RL_s = L_inductance_H / R_resistance_ohm; return {"tau_RL_s": tau_RL_s}
def calc_time_constant_RC_circuit(R_resistance_ohm, C_capacitance_F):
    if R_resistance_ohm <0 or C_capacitance_F <0 : return {"error":"电阻R和电容C不能为负"}
    tau_RC_s = R_resistance_ohm * C_capacitance_F; return {"tau_RC_s": tau_RC_s}
def calc_single_slit_diffraction_minima(m_order, lambda_wavelength, a_slit_width):
    if lambda_wavelength <= 0 or a_slit_width <= 0: return {"error": "波长和缝宽必须为正。"}
    try: m_order_int = int(m_order)
    except: return {"error": "衍射级数m必须为整数。"}
    if m_order_int == 0 or m_order != m_order_int : return {"error": "衍射级数m必须为非零整数。"}
    if a_slit_width == 0: return {"error": "缝宽a不能为零。"}
    sin_theta = (m_order_int * lambda_wavelength) / a_slit_width
    if abs(sin_theta) > 1: return {"error": f"sin(θ) = {sin_theta:.4f} > 1, 不存在此级暗纹。"}
    theta_rad = math.asin(sin_theta); theta_deg = math.degrees(theta_rad)
    return {"theta_deg_minima": theta_deg, "sin_theta_minima": sin_theta}
def calc_double_slit_interference_maxima(m_order, lambda_wavelength, d_slit_separation):
    if lambda_wavelength <= 0 or d_slit_separation <= 0: return {"error": "波长和缝间距必须为正。"}
    try: m_order_int = int(m_order)
    except: return {"error": "干涉级数m必须为整数。"}
    if m_order != m_order_int : return {"error": "干涉级数m必须为整数。"}
    if d_slit_separation == 0: return {"error": "缝间距d不能为零。"}
    sin_theta = (m_order_int * lambda_wavelength) / d_slit_separation
    if abs(sin_theta) > 1: return {"error": f"sin(θ) = {sin_theta:.4f} > 1, 不存在此级明纹。"}
    theta_rad = math.asin(sin_theta); theta_deg = math.degrees(theta_rad)
    return {"theta_deg_maxima": theta_deg, "sin_theta_maxima": sin_theta}
def calc_fluid_continuity(A1_area_m2, v1_velocity_ms, A2_area_m2):
    if A1_area_m2 <= 0 or v1_velocity_ms < 0 or A2_area_m2 <= 0: return {"error": "面积必须为正, 速度不能为负。"}
    if A2_area_m2 == 0: return {"error": "截面2面积A2不能为零。"}
    v2_velocity_ms = (A1_area_m2 * v1_velocity_ms) / A2_area_m2; return {"v2_velocity_ms": v2_velocity_ms}
def calc_first_law_thermo_deltaU(Q_heat_J, W_work_done_by_system_J):
    delta_U_internal_energy_J = Q_heat_J - W_work_done_by_system_J; return {"delta_U_internal_energy_J": delta_U_internal_energy_J}
def calc_rlc_series_impedance(R_ohm, XL_ohm_inductive_reactance, XC_ohm_capacitive_reactance):
    if R_ohm < 0 or XL_ohm_inductive_reactance < 0 or XC_ohm_capacitive_reactance < 0: return {"error": "电阻、感抗、容抗不能为负。"}
    Z_impedance_ohm = math.sqrt(R_ohm**2 + (XL_ohm_inductive_reactance - XC_ohm_capacitive_reactance)**2); return {"Z_impedance_ohm": Z_impedance_ohm}
def calc_transformer_voltage_turns(Vp_primary_V, Np_primary_turns, Ns_secondary_turns):
    if Vp_primary_V < 0 or Ns_secondary_turns < 0: return {"error": "电压和次级匝数不能为负"}
    try: Np_int = int(Np_primary_turns); Ns_int = int(Ns_secondary_turns)
    except: return {"error":"匝数必须为整数。"}
    if Np_primary_turns != Np_int or Ns_secondary_turns != Ns_int or Np_int <=0 : return {"error":"初级匝数必须为正整数, 次级匝数为整数。"}
    Vs_secondary_V = Vp_primary_V * (Ns_int / Np_int); return {"Vs_secondary_V": Vs_secondary_V}
def calc_stefan_boltzmann_law_power(emissivity, area_m2, T_kelvin):
    if not (0 <= emissivity <= 1) or area_m2 < 0 or T_kelvin < 0: return {"error": "发射率0-1, 面积和温度不能为负。"}
    P_radiated_power_W = emissivity * STEFAN_BOLTZMANN_CONSTANT_sigma * area_m2 * (T_kelvin**4); return {"P_radiated_power_W": P_radiated_power_W}
def calc_wien_displacement_law_lambda_max(T_kelvin):
    if T_kelvin <= 0: return {"error": "绝对温度T必须为正"}
    lambda_max_m = WIEN_DISPLACEMENT_CONSTANT_b / T_kelvin; return {"lambda_max_m": lambda_max_m}
def calc_lc_oscillation_frequency(L_inductance_H, C_capacitance_F):
    if L_inductance_H <= 0 or C_capacitance_F <= 0: return {"error": "电感L和电容C必须为正"}
    frequency_Hz = 1.0 / (2 * math.pi * math.sqrt(L_inductance_H * C_capacitance_F)); return {"frequency_Hz_LC": frequency_Hz}
def calc_round_aperture_diffraction_angle(lambda_wavelength, D_aperture_diameter):
    if lambda_wavelength <=0 or D_aperture_diameter <=0 : return {"error":"波长λ和孔径D必须为正。"}
    sin_theta_airy = 1.22 * (lambda_wavelength / D_aperture_diameter)
    if abs(sin_theta_airy) > 1: return {"error": f"sin(θ) = {sin_theta_airy:.4f} > 1，参数不适用或超出物理范围。"}
    theta_rad_airy = math.asin(sin_theta_airy); theta_deg_airy = math.degrees(theta_rad_airy)
    return {"theta_deg_airy_radius": theta_deg_airy, "sin_theta_airy": sin_theta_airy}
# NEW Physics Formulas
def calc_kinetic_friction(mu_k, N_normal_force):
    if mu_k < 0 or N_normal_force < 0: return {"error": "摩擦系数和正压力不能为负。"}
    F_k = mu_k * N_normal_force
    return {"F_kinetic_friction_N": F_k}

def calc_static_friction_max(mu_s, N_normal_force):
    if mu_s < 0 or N_normal_force < 0: return {"error": "摩擦系数和正压力不能为负。"}
    F_s_max = mu_s * N_normal_force
    return {"F_static_friction_max_N": F_s_max}

def calc_momentum(m_mass, v_velocity):
    if m_mass < 0: return {"error": "质量不能为负。"}
    p = m_mass * v_velocity
    return {"momentum_p": p}

def calc_impulse(F_force, delta_t_time):
    if delta_t_time < 0: return {"error": "时间变化不能为负。"}
    I = F_force * delta_t_time
    return {"impulse_I": I}

def calc_pressure_ideal_gas_kinetic_theory(N_particles, V_volume, KE_avg_particle):
    if N_particles < 0 or V_volume <= 0 or KE_avg_particle < 0: return {"error": "粒子数不能为负，体积需为正，平均动能不能为负。"}
    P = (2/3) * (N_particles / V_volume) * KE_avg_particle
    return {"pressure_Pa_kinetic": P}

def calc_rms_velocity_ideal_gas(M_molar_mass_kg_mol, T_kelvin):
    if M_molar_mass_kg_mol <= 0 or T_kelvin <= 0: return {"error": "摩尔质量和绝对温度必须为正。"}
    v_rms = math.sqrt((3 * R_GAS_CONSTANT * T_kelvin) / M_molar_mass_kg_mol)
    return {"v_rms_ms": v_rms}

def calc_electric_field_point_charge(q_charge_C, r_distance_m):
    if r_distance_m <= 0: return {"error": "距离r必须为正。"}
    E_field = K_COULOMB * q_charge_C / (r_distance_m**2)
    return {"E_field_N_C": E_field}

def calc_electric_potential_point_charge(q_charge_C, r_distance_m):
    if r_distance_m <= 0: return {"error": "距离r必须为正。"}
    V_potential = K_COULOMB * q_charge_C / r_distance_m
    return {"V_potential_V": V_potential}

def calc_magnetic_field_straight_wire(I_current_A, r_distance_m):
    if I_current_A < 0 or r_distance_m <= 0: return {"error": "电流不能为负，距离需为正。"}
    B_field = (PERMEABILITY_FREE_SPACE_mu0 * I_current_A) / (2 * math.pi * r_distance_m)
    return {"B_field_T": B_field}

def calc_induced_emf_faradays_law(delta_phi_magnetic_flux_Wb, delta_t_time_s):
    if delta_t_time_s <= 0: return {"error": "时间变化量必须为正。"}
    emf = -delta_phi_magnetic_flux_Wb / delta_t_time_s
    return {"emf_induced_V": emf}

def calc_energy_stored_inductor(L_inductance_H, I_current_A):
    if L_inductance_H < 0: return {"error": "电感不能为负。"}
    U_inductor = 0.5 * L_inductance_H * (I_current_A**2)
    return {"U_inductor_J": U_inductor}

def calc_power_in_AC_circuit(V_rms_V, I_rms_A, cos_phi_power_factor):
    if V_rms_V < 0 or I_rms_A < 0 or not (0 <= cos_phi_power_factor <= 1): return {"error": "电压、电流不能为负，功率因数需在0-1之间。"}
    P_avg = V_rms_V * I_rms_A * cos_phi_power_factor
    return {"P_avg_watt": P_avg}

def calc_capacitive_reactance(f_frequency_Hz, C_capacitance_F):
    if f_frequency_Hz < 0 or C_capacitance_F < 0: return {"error": "频率和电容不能为负。"}
    if f_frequency_Hz == 0 or C_capacitance_F == 0: return {"XC_ohm": float('inf'), "info": "频率或电容为零，容抗无限大。"}
    Xc = 1 / (2 * math.pi * f_frequency_Hz * C_capacitance_F)
    return {"XC_ohm": Xc}

def calc_inductive_reactance(f_frequency_Hz, L_inductance_H):
    if f_frequency_Hz < 0 or L_inductance_H < 0: return {"error": "频率和电感不能为负。"}
    Xl = 2 * math.pi * f_frequency_Hz * L_inductance_H
    return {"XL_ohm": Xl}

def calc_resonance_frequency_RLC(L_inductance_H, C_capacitance_F):
    if L_inductance_H <= 0 or C_capacitance_F <= 0: return {"error": "电感和电容必须为正。"}
    f_res = 1 / (2 * math.pi * math.sqrt(L_inductance_H * C_capacitance_F))
    return {"f_resonance_Hz": f_res}

def calc_photon_energy_from_wavelength_nm(lambda_nm):
    if lambda_nm <= 0: return {"error": "波长必须为正。"}
    lambda_m = lambda_nm * 1e-9
    E_photon_J = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / lambda_m
    E_photon_eV = E_photon_J / ELEMENTARY_CHARGE
    return {"E_photon_J": E_photon_J, "E_photon_eV": E_photon_eV}


# Chemistry
def calc_ideal_gas_pressure(n, temp_c, vol_L):
    if n < 0 or vol_L <= 0 : return {"error": "物质的量n不能为负, 体积V必须为正。"}
    temp_k = temp_c + 273.15
    if temp_k <=0: return {"error": "绝对温度T必须为正"}
    vol_m3 = vol_L / 1000.0; pressure_Pa = (n * R_GAS_CONSTANT * temp_k) / vol_m3
    return {"P_Pa": pressure_Pa}
def calc_molarity_M(moles, vol_L):
    if moles < 0 or vol_L <= 0 : return {"error": "物质的量不能为负, 溶液体积V必须为正。"}
    molarity = moles / vol_L; return {"M": molarity}
def calc_ph_from_h(h_plus_conc):
    if h_plus_conc <= 0: return {"error": "[H⁺]浓度必须为正"}
    try: ph = -math.log10(h_plus_conc)
    except ValueError: return {"error":"[H⁺]的无效值用于对数计算."}
    return {"pH": ph}
def calc_percent_yield(actual_yield, theoretical_yield):
    if actual_yield < 0 or theoretical_yield <= 0 : return {"error": "产率不能为负, 理论产率需为正。"} # If theoretical is 0, py is undefined/inf
    if theoretical_yield == 0 and actual_yield !=0: return {"error":"理论产率为零但实际产率非零。"}
    if theoretical_yield == 0 and actual_yield ==0: return {"percent_yield": 0.0} # Or undefined
    py = (actual_yield / theoretical_yield) * 100; return {"percent_yield": py}
def calc_dilution_m2(m1, v1, v2):
    if v1 < 0 or v2 <= 0 or m1 <0: return {"error":"体积需为正, 浓度不能为负。"}
    m2 = (m1 * v1) / v2; return {"M2": m2}
def calc_dilution_v2(m1, v1, m2):
    if v1 < 0 or m1 <0 or m2 <=0: return {"error":"初始体积/浓度不能为负, 最终浓度M2需为正。"}
    if m1 == 0: return {"error": "无法从M1=0通过稀释得到M2>0。"} if m2 > 0 else {"V2":v1} # M2=0 also from M1=0
    v2 = (m1 * v1) / m2
    if v2 < v1 and m2 > m1: return {"V2":v2, "info": "注意: M2 > M1 需要浓缩 (V2 < V1), 而非稀释."}
    return {"V2": v2}
def calc_gibbs_free_energy(delta_H_kJ, temp_c, delta_S_J_K):
    delta_H_J = delta_H_kJ * 1000; temp_K = temp_c + 273.15
    if temp_K <= 0: return {"error": "绝对温度T必须为正"}
    delta_G_J = delta_H_J - (temp_K * delta_S_J_K); delta_G_kJ = delta_G_J / 1000
    return {"delta_G_kJ": delta_G_kJ}
def calc_half_life_first_order(k):
    if k <= 0: return {"error": "速率常数k必须为正"}
    t_half = math.log(2) / k; return {"t_half": t_half}
def calc_arrhenius_k(A, Ea_kJ_mol, temp_c):
    if A <0 : return {"error":"指前因子A不能为负"}
    Ea_J_mol = Ea_kJ_mol * 1000; temp_K = temp_c + 273.15
    if temp_K <= 0: return {"error": "绝对温度T必须为正"}
    try: k = A * math.exp(-Ea_J_mol / (R_GAS_CONSTANT * temp_K))
    except OverflowError: return {"error":"阿伦尼乌斯计算溢出. Ea/(RT)可能过大或过小."}
    return {"k_rate_const": k}
def calc_moles_from_mass(mass_g, molar_mass_g_mol):
    if mass_g < 0 or molar_mass_g_mol <= 0 : return {"error":"质量不能为负, 摩尔质量需为正。"}
    n_moles = mass_g / molar_mass_g_mol; return {"n_moles": n_moles}
def calc_mass_percent(mass_solute_g, mass_solution_g):
    if mass_solute_g < 0 or mass_solution_g <= 0: return {"error":"质量不能为负, 溶液质量需为正。"}
    if mass_solution_g == 0 and mass_solute_g !=0: return {"error":"溶液质量为零但溶质质量非零。"}
    if mass_solution_g == 0 and mass_solute_g ==0: return {"mass_percent": 0.0}
    if mass_solute_g > mass_solution_g : return {"error": "溶质质量不应大于溶液质量."}
    percent_mass = (mass_solute_g / mass_solution_g) * 100; return {"mass_percent": percent_mass}
def calc_molality(moles_solute, mass_solvent_kg):
    if moles_solute <0 or mass_solvent_kg <=0: return {"error":"物质的量不能为负, 溶剂质量需为正。"}
    if mass_solvent_kg == 0 and moles_solute !=0: return {"error": "溶剂质量(kg)为零但溶质的物质的量非零."}
    if mass_solvent_kg == 0 and moles_solute ==0: return {"m_molality":0.0}
    m_molality = moles_solute / mass_solvent_kg; return {"m_molality": m_molality}
def calc_nernst_equation_Ecell(E_standard_cell_V, n_electrons, Q_reaction_quotient, temp_c=25):
    try: n_val = float(n_electrons)
    except: return {"error":"转移电子数n必须是数字。"}
    if not (n_val.is_integer() and n_val > 0): return {"error":"转移电子数n必须为正整数。"}
    n_electrons_int = int(n_val)
    if Q_reaction_quotient <= 0: return {"error": "反应商Q必须为正"}
    temp_K = temp_c + 273.15;
    if temp_K <=0: return {"error":"绝对温度T必须为正"}
    nernst_factor_ln = (R_GAS_CONSTANT * temp_K) / (n_electrons_int * FARADAY_CONSTANT_DERIVED)
    try: E_cell = E_standard_cell_V - nernst_factor_ln * math.log(Q_reaction_quotient)
    except ValueError: return {"error": "能斯特方程对数计算中Q的无效值."}
    except OverflowError: return {"error": "能斯特方程计算溢出. (RT/nF)ln(Q)项可能过大."}
    return {"E_cell_V": E_cell}
def calc_henderson_hasselbalch_ph(pKa, A_minus_conc, HA_conc):
    if A_minus_conc < 0 or HA_conc < 0 : return {"error": "浓度不能为负"}
    if abs(HA_conc) < 1e-9 and abs(A_minus_conc) < 1e-9: return {"error": "[HA]和[A⁻]不能同时为零."}
    if abs(HA_conc) < 1e-9 : return {"error": "[HA]为零但[A⁻]非零时, 比值未定义."}
    effective_A_minus_conc = max(A_minus_conc, 1e-100) # Avoid direct log(0)
    ratio = effective_A_minus_conc / HA_conc
    if ratio <= 0 : return {"error": "比值[A⁻]/[HA]必须为正才能进行对数计算."}
    try: pH = pKa + math.log10(ratio)
    except ValueError: return {"error":"计算比值[A⁻]/[HA]的对数时出错."}
    return {"pH_buffer": pH}
def calc_radioactive_decay_Nt(N0_initial_amount, time_elapsed, half_life):
    if N0_initial_amount <0 or time_elapsed <0 or half_life <=0 : return {"error":"初始量/时间不能为负, 半衰期T½需为正。"}
    try: Nt_remaining = N0_initial_amount * (0.5)**(time_elapsed / half_life)
    except OverflowError: return {"error":"放射性衰变计算溢出. (t/T½) 可能过大."}
    return {"Nt_remaining": Nt_remaining}
def calc_ideal_gas_volume(n_moles, temp_c, pressure_Pa): # Duplicate? Keep only one.
    if n_moles < 0 or pressure_Pa <=0: return {"error": "物质的量n不能为负, 压强P必须为正。"}
    temp_K = temp_c + 273.15;
    if temp_K <=0: return {"error":"绝对温度T必须为正"}
    V_m3 = (n_moles * R_GAS_CONSTANT * temp_K) / pressure_Pa; V_L = V_m3 * 1000
    return {"V_liters": V_L, "V_cubic_meters": V_m3}
def calc_ideal_gas_moles(pressure_Pa, vol_L, temp_c): # Duplicate?
    if pressure_Pa <=0 or vol_L <=0 : return {"error":"压强P或体积V必须为正"}
    temp_K = temp_c + 273.15
    if temp_K <=0: return {"error":"绝对温度T必须为正"}
    if abs(R_GAS_CONSTANT * temp_K) < 1e-9: return {"error":"RT乘积为零, 无法相除."}
    vol_m3 = vol_L / 1000.0; n_moles = (pressure_Pa * vol_m3) / (R_GAS_CONSTANT * temp_K)
    return {"n_moles_calculated": n_moles}
def calc_ideal_gas_temp(pressure_Pa, vol_L, n_moles): # Duplicate?
    if pressure_Pa <=0 or vol_L <=0 or n_moles <=0 : return {"error":"压强P, 体积V或物质的量n必须为正"}
    if abs(n_moles * R_GAS_CONSTANT) < 1e-9: return {"error": "nR乘积为零, 无法相除."}
    vol_m3 = vol_L / 1000.0; T_Kelvin = (pressure_Pa * vol_m3) / (n_moles * R_GAS_CONSTANT)
    if T_Kelvin < 0: return {"error":"计算的绝对温度为负, 检查输入."}
    T_Celsius = T_Kelvin - 273.15; return {"T_Kelvin": T_Kelvin, "T_Celsius": T_Celsius}
def calc_colligative_temp_change(i_vant_hoff, K_molal_constant, m_molality): # Duplicate?
    if i_vant_hoff < 0 or K_molal_constant < 0 or m_molality < 0: return {"error":"输入参数不能为负"}
    delta_T = i_vant_hoff * K_molal_constant * m_molality
    return {"delta_T_change": delta_T}
def calc_gas_moles_at_STP(volume_L_at_STP): # Duplicate?
    if volume_L_at_STP < 0 : return {"error":"STP下气体体积V_STP不能为负"}
    if abs(STP_MOLAR_VOLUME) < 1e-9 : return {"error":"STP摩尔体积常数为零."}
    n_moles = volume_L_at_STP / STP_MOLAR_VOLUME; return {"n_moles_STP": n_moles}
def calc_number_of_particles(n_moles): # Duplicate?
    if n_moles < 0: return {"error": "物质的量n不能为负"}
    N_particles = n_moles * AVOGADRO_CONSTANT; return {"N_particles": N_particles}
def calc_molar_solubility_s_from_Ksp_AB(Ksp_AB): # Duplicate?
    if Ksp_AB < 0: return {"error": "溶度积Ksp不能为负"}
    s_molar_solubility = math.sqrt(Ksp_AB); return {"s_molar_solubility": s_molar_solubility}
def calc_molar_solubility_s_from_Ksp_A2B_or_AB2(Ksp_A2B_or_AB2): # Duplicate?
    if Ksp_A2B_or_AB2 < 0: return {"error": "溶度积Ksp不能为负"}
    if abs(Ksp_A2B_or_AB2) < 1e-18 : return {"s_molar_solubility": 0.0}
    val_for_cbrt = Ksp_A2B_or_AB2 / 4.0
    if val_for_cbrt < 0: return {"error":"Ksp/4 小于0, 无法计算实数立方根."}
    s_molar_solubility = val_for_cbrt**(1.0/3.0); return {"s_molar_solubility": s_molar_solubility}
def calc_pH_from_pOH(pOH, temp_c=25.0): # Duplicate?
    if abs(temp_c - 25.0) > 1e-3 : return {"error": "此简化换算pH+pOH=14仅对25°C有效."}
    pH = 14.0 - pOH; return {"pH_calculated": pH}
def calc_pOH_from_pH(pH, temp_c=25.0): # Duplicate?
    if abs(temp_c - 25.0) > 1e-3 : return {"error": "此简化换算pH+pOH=14仅对25°C有效."}
    pOH = 14.0 - pH; return {"pOH_calculated": pOH}
def calc_solution_freezing_point(delta_Tf_depression, Tf_pure_solvent_C = PURE_WATER_FREEZING_POINT_C ): # Ensure parameters match ALL_FORMULAS if default used for first
    if delta_Tf_depression < 0: return {"error":"凝固点降低值ΔTf应为正或零."}
    Tf_solution_C = Tf_pure_solvent_C - delta_Tf_depression; return {"Tf_solution_C": Tf_solution_C}
def calc_solution_boiling_point(delta_Tb_elevation, Tb_pure_solvent_C = PURE_WATER_BOILING_POINT_C ): # Same here for param order
    if delta_Tb_elevation < 0: return {"error":"沸点升高值ΔTb应为正或零."}
    Tb_solution_C = Tb_pure_solvent_C + delta_Tb_elevation; return {"Tb_solution_C": Tb_solution_C}
def calc_henrys_law_solubility(kH_henry_const_M_atm, P_partial_pressure_gas_atm):
    if kH_henry_const_M_atm < 0 or P_partial_pressure_gas_atm < 0: return {"error": "亨利常数kH和气体分压P不能为负。"}
    S_solubility_M = kH_henry_const_M_atm * P_partial_pressure_gas_atm; return {"S_solubility_M": S_solubility_M}
def calc_Kp_from_Kc(Kc_equilibrium_const_conc, R_gas_const_L_atm_mol_K, T_kelvin, delta_n_moles_gas):
    if Kc_equilibrium_const_conc < 0 or R_gas_const_L_atm_mol_K <= 0 or T_kelvin < 0: return {"error": "Kc/T不能为负, R需为正。"}
    try: Kp_val = Kc_equilibrium_const_conc * (R_gas_const_L_atm_mol_K * T_kelvin)**(delta_n_moles_gas)
    except OverflowError: return {"error":"计算Kp时发生溢出."}
    except ValueError: return {"error":"计算Kp时发生数值错误."}
    return {"Kp_equilibrium_const_pressure": Kp_val}
def calc_bohr_model_energy_transition(ni_initial_level, nf_final_level):
    try: ni = int(ni_initial_level); nf = int(nf_final_level)
    except: return {"error":"能级ni,nf必须为整数。"}
    if ni <=0 or nf <=0 or ni == nf: return {"error":"能级ni,nf必须为不同正整数。"}
    RH_energy_J = 2.17987236e-18; delta_E_J = -RH_energy_J * ((1/(nf**2)) - (1/(ni**2)))
    wavelength_m = 0; frequency_Hz = 0
    if abs(delta_E_J) > 1e-30: wavelength_m = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / abs(delta_E_J); frequency_Hz = abs(delta_E_J) / PLANCK_CONSTANT
    return {"delta_E_J": delta_E_J, "photon_energy_J_abs": abs(delta_E_J), "photon_wavelength_m": wavelength_m, "photon_frequency_Hz": frequency_Hz, "transition_type": "发射" if delta_E_J < 0 else "吸收"}
def calc_raoults_law_vapor_pressure_solution(P0_solvent_atm, X_solvent_mole_fraction):
    if P0_solvent_atm < 0 or not (0 <= X_solvent_mole_fraction <= 1): return {"error": "纯溶剂蒸气压P°不能为负, 摩尔分数X在0-1间。"}
    P_solution_atm = X_solvent_mole_fraction * P0_solvent_atm; return {"P_solution_atm": P_solution_atm}
def calc_degree_of_dissociation_weak_acid(Ka_ionization_const, C0_initial_conc_M):
    if Ka_ionization_const < 0 or C0_initial_conc_M <= 0: return {"error": "Ka不能为负, 初始浓度C₀需为正。"}
    discriminant = Ka_ionization_const**2 + 4 * C0_initial_conc_M * Ka_ionization_const
    if discriminant < 0 : return {"error":"解离度计算判别式为负。"}
    alpha_plus = (-Ka_ionization_const + math.sqrt(discriminant)) / (2 * C0_initial_conc_M)
    if 0 <= alpha_plus <= 1: alpha = alpha_plus
    else:
        alpha_approx = math.sqrt(Ka_ionization_const / C0_initial_conc_M) if C0_initial_conc_M > 0 else float('inf')
        if 0 <= alpha_approx <= 0.05: alpha = alpha_approx; return {"alpha_dissociation_fraction": alpha, "alpha_percent": alpha * 100, "info":"使用了近似公式α=√(Ka/C₀)"}
        else: alpha = alpha_plus; return {"alpha_dissociation_fraction": alpha, "alpha_percent": alpha * 100, "info":"注意: α > 0.05 或超出 [0,1], 近似可能较差或为强电解质."}
    return {"alpha_dissociation_fraction": alpha, "alpha_percent": alpha * 100}
def calc_ionic_product_water_Kw_approx_temp(temp_c):
    Kw_val = 0
    if abs(temp_c - 0.0) < 1e-3: Kw_val = 1.139e-15
    elif abs(temp_c - 25.0) < 1e-3: Kw_val = 1.008e-14
    elif abs(temp_c - 60.0) < 1e-3: Kw_val = 9.614e-14
    elif abs(temp_c - 100.0) < 1e-3: Kw_val = 5.47e-13
    else: return {"error": "此温度的Kw值需查表或复杂公式. 提供了0,25,60,100°C参考点."}
    pKw = -math.log10(Kw_val) if Kw_val > 0 else float('inf')
    return {"Kw_ionic_product": Kw_val, "pKw": pKw, "temp_used_C": temp_c}

# NEW Chemistry Formulas
def calc_half_life_zero_order(A0_initial_conc_M, k_rate_constant_M_s):
    if A0_initial_conc_M < 0 or k_rate_constant_M_s <= 0: return {"error": "初始浓度不能为负，速率常数k必须为正。"}
    t_half = A0_initial_conc_M / (2 * k_rate_constant_M_s)
    return {"t_half_zero_order_s": t_half}

def calc_half_life_second_order(A0_initial_conc_M, k_rate_constant_M_inv_s_inv):
    if A0_initial_conc_M <= 0 or k_rate_constant_M_inv_s_inv <= 0: return {"error": "初始浓度和速率常数k必须为正。"}
    t_half = 1 / (k_rate_constant_M_inv_s_inv * A0_initial_conc_M)
    return {"t_half_second_order_s": t_half}

def calc_faradays_law_mass_electrolysis(current_A, time_s, M_molar_mass_g_mol, n_electrons_transferred):
    if current_A < 0 or time_s < 0 or M_molar_mass_g_mol <= 0 or n_electrons_transferred <= 0: return {"error": "电流、时间不能为负，摩尔质量和转移电子数需为正。"}
    try: n_electrons_int = int(n_electrons_transferred)
    except: return {"error":"转移电子数n必须是整数。"}
    if n_electrons_transferred != n_electrons_int or n_electrons_int <=0 : return {"error":"转移电子数n必须为正整数。"}

    charge_C = current_A * time_s
    moles_produced = charge_C / (n_electrons_int * FARADAY_CONSTANT_DERIVED)
    mass_deposited_g = moles_produced * M_molar_mass_g_mol
    return {"mass_deposited_g": mass_deposited_g, "moles_produced": moles_produced}

def calc_osmotic_pressure_MRT(M_molarity_mol_L, T_kelvin, R_gas_const_L_atm_mol_K=GAS_CONSTANT_L_ATM, i_vant_hoff=1.0):
    if M_molarity_mol_L < 0 or T_kelvin < 0 or R_gas_const_L_atm_mol_K <= 0 or i_vant_hoff < 0: return {"error": "浓度、温度、气体常数、范特霍夫因子不能为负，R需为正。"}
    osmotic_pressure_atm = i_vant_hoff * M_molarity_mol_L * R_gas_const_L_atm_mol_K * T_kelvin
    osmotic_pressure_Pa = osmotic_pressure_atm * STANDARD_ATM_PRESSURE_Pa
    return {"osmotic_pressure_atm": osmotic_pressure_atm, "osmotic_pressure_Pa": osmotic_pressure_Pa}

def calc_heat_of_reaction_from_enthalpies_of_formation(stoich_coeffs, delta_Hf_products_kJ_mol, delta_Hf_reactants_kJ_mol):
    # stoich_coeffs is a dictionary like {"product1": coeff1, "reactant1": -coeff2}
    # delta_Hf_products_kJ_mol and delta_Hf_reactants_kJ_mol are dictionaries like {"product1": hf1, "reactant1": hf2}
    # Ensure coefficients for products are positive, reactants are negative
    if not all(isinstance(c, (int, float)) for c in stoich_coeffs.values()): return {"error": "化学计量数必须是数字。"}
    if not all(isinstance(h, (int, float)) for h in delta_Hf_products_kJ_mol.values()): return {"error": "生成焓必须是数字。"}
    if not all(isinstance(h, (int, float)) for h in delta_Hf_reactants_kJ_mol.values()): return {"error": "生成焓必须是数字。"}

    sum_hf_products = sum(stoich_coeffs.get(p, 0) * hf for p, hf in delta_Hf_products_kJ_mol.items())
    sum_hf_reactants = sum(abs(stoich_coeffs.get(r, 0)) * hf for r, hf in delta_Hf_reactants_kJ_mol.items()) # Use abs for reactants

    delta_H_rxn_kJ_mol = sum_hf_products - sum_hf_reactants
    return {"delta_H_rxn_kJ_mol": delta_H_rxn_kJ_mol}

def calc_equilibrium_constant_from_gibbs_free_energy(delta_G_standard_kJ_mol, temp_c):
    delta_G_standard_J_mol = delta_G_standard_kJ_mol * 1000
    temp_K = temp_c + 273.15
    if temp_K <= 0: return {"error": "绝对温度T必须为正。"}
    try:
        # -RT ln K = ΔG°
        # ln K = -ΔG° / (RT)
        # K = exp(-ΔG° / (RT))
        exponent = -delta_G_standard_J_mol / (R_GAS_CONSTANT * temp_K)
        K_eq = math.exp(exponent)
    except OverflowError: return {"error": "平衡常数计算溢出，ΔG°/(RT)可能过大。"}
    except ValueError: return {"error": "平衡常数计算错误，可能是对负数取对数。"}
    return {"K_equilibrium_constant": K_eq}

def calc_van_der_waals_pressure(n_moles, V_volume_L, T_celsius, a_vdw, b_vdw):
    if n_moles <= 0 or V_volume_L <= 0 or a_vdw < 0 or b_vdw < 0: return {"error": "物质的量、体积需为正，a和b不能为负。"}
    T_kelvin = T_celsius + 273.15
    if T_kelvin <= 0: return {"error": "绝对温度T必须为正。"}
    V_m3 = V_volume_L / 1000.0
    V_molar_m3 = V_m3 / n_moles if n_moles > 0 else float('inf')

    # (P + a(n/V)²)(V/n - b) = RT
    # P = (RT / (V/n - b)) - a(n/V)²
    try:
        term1_denominator = V_molar_m3 - b_vdw
        if abs(term1_denominator) < 1e-9: return {"error": "范德华方程计算错误: Vm - b 接近零。"}
        term1 = (R_GAS_CONSTANT * T_kelvin) / term1_denominator
        term2 = a_vdw * (n_moles / V_m3)**2
        P_vdw_Pa = term1 - term2
    except OverflowError: return {"error": "范德华方程计算溢出。"}
    except ValueError: return {"error": "范德华方程计算错误。"}

    return {"P_van_der_waals_Pa": P_vdw_Pa}

def calc_heat_capacity_constant_volume(delta_U_J, delta_T_K):
    if delta_T_K == 0: return {"error": "温度变化ΔT不能为零。"}
    Cv = delta_U_J / delta_T_K
    return {"Cv_J_K": Cv}

def calc_heat_capacity_constant_pressure(delta_H_J, delta_T_K):
    if delta_T_K == 0: return {"error": "温度变化ΔT不能为零。"}
    Cp = delta_H_J / delta_T_K
    return {"Cp_J_K": Cp}


# Biology
def calc_population_growth_exponential(N0, r_rate, t_time):
    if N0 < 0 or t_time < 0 : return {"error": "初始数量N0或时间t不能为负"}
    try: Nt = N0 * math.exp(r_rate * t_time)
    except OverflowError: return {"error":"指数增长计算溢出. r*t可能过大."}
    return {"Nt": Nt}
def calc_hardy_weinberg_from_q2(q_squared):
    if not (0 <= q_squared <= 1): return {"error": "q²必须在0和1之间"}
    q = math.sqrt(q_squared); p = 1.0 - q; p_squared = p**2; two_pq = 2 * p * q
    return {"p_A_freq": p, "q_a_freq": q, "p_squared_AA": p_squared, "two_pq_Aa": two_pq, "q_squared_aa_input": q_squared}
def calc_magnification(image_size, actual_size):
    if image_size <0 or actual_size <=0 : return {"error": "尺寸不能为负, 实际大小需为正。"}
    mag = image_size / actual_size; return {"magnification": mag}
def calc_bmi(weight_kg, height_cm):
    if weight_kg <= 0 or height_cm <= 0: return {"error": "体重和身高必须为正"}
    height_m = height_cm / 100.0; bmi = weight_kg / (height_m ** 2)
    category = "未知";
    if bmi < 18.5: category = "体重过轻"
    elif 18.5 <= bmi < 24: category = "体重正常 (中国标准)"
    elif 24 <= bmi < 28: category = "超重 (中国标准)"
    else: category = "肥胖 (中国标准)"
    return {"BMI": bmi, "Category": category}
def calc_logistic_growth_Nt(N0, K_capacity, r_rate, t_time):
    if N0 <= 0 or K_capacity <= 0 or t_time <0: return {"error": "N0,K需为正, t不能为负。"}
    if abs(N0 - K_capacity) < 1e-9: return {"Nt_logistic": K_capacity}
    try:
        if abs(N0) < 1e-9 : return {"error":"N0为零, 公式项无效."}
        ratio_term = (K_capacity - N0) / N0; exp_term = math.exp(-r_rate * t_time)
        Nt = K_capacity / (1 + ratio_term * exp_term)
    except OverflowError: return {"error": "逻辑斯谛增长计算溢出."}
    return {"Nt_logistic": Nt}
def calc_michaelis_menten_V(Vmax, S_conc, Km):
    if Vmax <0 or S_conc <0 or Km <0: return {"error":"Vmax, [S]或Km不能为负"}
    denominator = Km + S_conc
    if abs(denominator) < 1e-9 : return {"error": "Km + [S]为零, 导致除零."} if abs(Vmax*S_conc) > 1e-9 else {"V_reaction_rate":0.0}
    V_reaction_rate = (Vmax * S_conc) / denominator; return {"V_reaction_rate": V_reaction_rate}
def calc_mark_recapture_N(M_marked_first, C_captured_second, R_recaptured_marked):
    if M_marked_first <0 or C_captured_second <0 or R_recaptured_marked <=0: return {"error": "计数不能为负, R需为正。"}
    if R_recaptured_marked > C_captured_second or R_recaptured_marked > M_marked_first: return {"error": "R不应大于C或M."}
    N_population_estimate = (M_marked_first * C_captured_second) / R_recaptured_marked; return {"N_population_estimate": N_population_estimate}
def calc_photosynthesis_respiration_rate(gas_change_volume_mL, time_min, biomass_g, molar_mass_gas_g_mol=MOLAR_MASS_O2_g_mol):
    if gas_change_volume_mL < 0 or time_min <= 0 or biomass_g <= 0 or molar_mass_gas_g_mol <= 0 : return {"error":"输入参数需为正(气体体积可为0)。"}
    if abs(STP_MOLAR_VOLUME) < 1e-9: return {"error":"STP摩尔体积常数为零"}
    moles_gas = (gas_change_volume_mL / 1000.0) / STP_MOLAR_VOLUME
    rate_umol_g_min = (moles_gas * 1e6) / (biomass_g * time_min) if biomass_g * time_min != 0 else float('inf')
    return {"rate_umol_g_min": rate_umol_g_min, "moles_gas_evolved_consumed": moles_gas}
def calc_lineweaver_burk_params(S_substrate_conc, V_initial_velocity):
    if S_substrate_conc <= 0 or V_initial_velocity <= 0: return {"error":"底物浓度[S]和初始速率V必须为正"}
    inv_S = 1.0 / S_substrate_conc; inv_V = 1.0 / V_initial_velocity
    return {"inv_S": inv_S, "inv_V": inv_V}
def calc_bmr_harris_benedict(weight_kg, height_cm, age_years, gender_male_bool):
    if weight_kg <=0 or height_cm <=0 or age_years <=0 : return {"error":"体重、身高、年龄必须为正"}
    if gender_male_bool: bmr_kcal_day = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) + 5
    else: bmr_kcal_day = (10 * weight_kg) + (6.25 * height_cm) - (5 * age_years) - 161
    return {"bmr_kcal_day_MifflinStJeor": bmr_kcal_day} # Note: This is Mifflin-St Jeor, not Harris-Benedict

def calc_cardiac_output(stroke_volume_mL_beat, heart_rate_bpm):
    if stroke_volume_mL_beat < 0 or heart_rate_bpm < 0: return {"error":"每搏输出量和心率不能为负。"}
    CO_L_min = (stroke_volume_mL_beat / 1000.0) * heart_rate_bpm
    return {"CO_L_min": CO_L_min}
def calc_respiratory_quotient(CO2_eliminated_moles, O2_consumed_moles):
    if CO2_eliminated_moles < 0 or O2_consumed_moles < 0: return {"error":"气体摩尔数不能为负。"}
    if O2_consumed_moles == 0: return {"RQ": float('nan'), "info":"O₂消耗量为零, RQ未定义。"} if CO2_eliminated_moles == 0 else {"RQ": float('inf'), "info":"O₂消耗量为零, RQ为无穷大."}
    RQ = CO2_eliminated_moles / O2_consumed_moles; return {"RQ": RQ}
def calc_max_heart_rate_estimate_tanaka(age_years):
    if age_years <= 0: return {"error": "估算最大心率的年龄必须为正。"}
    hr_max_bpm_tanaka = 208 - (0.7 * age_years); return {"hr_max_bpm_tanaka": hr_max_bpm_tanaka}
def calc_water_potential_osmotic(i_vant_hoff, C_molar_conc_mol_L, T_kelvin, R_gas_const_MPa_L_mol_K=GAS_CONSTANT_MPA_L):
    if i_vant_hoff < 0 or C_molar_conc_mol_L < 0 or T_kelvin < 0 or R_gas_const_MPa_L_mol_K <=0 : return {"error":"i,C,T不能为负, R需为正。"}
    psi_s_MPa = -i_vant_hoff * C_molar_conc_mol_L * R_gas_const_MPa_L_mol_K * T_kelvin
    return {"psi_s_osmotic_potential_MPa": psi_s_MPa}
def calc_doubling_time_from_growth_rate(r_specific_growth_rate_per_time):
    if r_specific_growth_rate_per_time <= 0: return {"error":"比增长速率r必须为正。"}
    t_doubling = math.log(2) / r_specific_growth_rate_per_time; return {"t_doubling": t_doubling}
def calc_competitive_inhibition_apparent_Km(Km_original_M, I_inhibitor_conc_M, Ki_inhibitor_const_M):
    if Km_original_M <0 or I_inhibitor_conc_M <0 or Ki_inhibitor_const_M <0: return {"error":"Km, [I], 或 Ki 不能为负。"}
    if Ki_inhibitor_const_M == 0: Km_app_M = Km_original_M if I_inhibitor_conc_M == 0 else float('inf') # Simplified Ki=0 implies very strong binding -> large effect if I>0
    else: Km_app_M = Km_original_M * (1 + (I_inhibitor_conc_M / Ki_inhibitor_const_M))
    return {"Km_apparent_M": Km_app_M}

# NEW Biology Formulas
def calc_population_density(population_size, area):
    if population_size < 0 or area <= 0: return {"error": "种群大小不能为负，面积需为正。"}
    density = population_size / area
    return {"population_density": density}

def calc_birth_rate(births, population_size, time_unit=1):
    if births < 0 or population_size <= 0 or time_unit <= 0: return {"error": "出生数不能为负，种群大小和时间单位需为正。"}
    birth_rate = (births / population_size) / time_unit
    return {"birth_rate_per_capita_per_time": birth_rate}

def calc_death_rate(deaths, population_size, time_unit=1):
    if deaths < 0 or population_size <= 0 or time_unit <= 0: return {"error": "死亡数不能为负，种群大小和时间单位需为正。"}
    death_rate = (deaths / population_size) / time_unit
    return {"death_rate_per_capita_per_time": death_rate}

def calc_population_growth_rate_birth_death(birth_rate, death_rate, population_size):
    if birth_rate < 0 or death_rate < 0 or population_size < 0: return {"error": "出生率、死亡率、种群大小不能为负。"}
    growth_rate = (birth_rate - death_rate) * population_size
    return {"population_growth_rate": growth_rate}

def calc_allele_frequency(count_of_allele, total_alleles):
    if count_of_allele < 0 or total_alleles <= 0: return {"error": "等位基因计数不能为负，总等位基因数需为正。"}
    if count_of_allele > total_alleles: return {"error": "等位基因计数不能大于总等位基因数。"}
    frequency = count_of_allele / total_alleles
    return {"allele_frequency": frequency}

def calc_genotype_frequency(count_of_genotype, total_individuals):
    if count_of_genotype < 0 or total_individuals <= 0: return {"error": "基因型计数不能为负，总个体数需为正。"}
    if count_of_genotype > total_individuals: return {"error": "基因型计数不能大于总个体数。"}
    frequency = count_of_genotype / total_individuals
    return {"genotype_frequency": frequency}

def calc_enzyme_efficiency_kcat_Km(kcat, Km):
    if kcat < 0 or Km <= 0: return {"error": "kcat不能为负，Km需为正。"}
    efficiency = kcat / Km
    return {"enzyme_efficiency_kcat_per_Km": efficiency}

def calc_hardy_weinberg_from_p_q(p_freq, q_freq):
    if not (0 <= p_freq <= 1) or not (0 <= q_freq <= 1): return {"error": "p和q必须在0和1之间。"}
    if abs(p_freq + q_freq - 1.0) > 1e-9: return {"error": "p + q 必须等于 1。"}
    p_squared = p_freq**2
    q_squared = q_freq**2
    two_pq = 2 * p_freq * q_freq
    return {"p_freq_input": p_freq, "q_freq_input": q_freq, "p_squared_AA": p_squared, "q_squared_aa": q_squared, "two_pq_Aa": two_pq}

def calc_percent_error(experimental_value, theoretical_value):
    if theoretical_value == 0:
        if experimental_value == 0: return {"percent_error": 0.0}
        else: return {"percent_error": float('inf'), "info": "理论值为零，误差百分比无限大。"}
    error = abs(experimental_value - theoretical_value) / abs(theoretical_value) * 100
    return {"percent_error": error}

def calc_dilution_factor(initial_volume, final_volume):
    if initial_volume <= 0 or final_volume <= 0: return {"error": "初始体积和最终体积必须为正。"}
    if final_volume < initial_volume: return {"error": "最终体积不能小于初始体积。"}
    dilution_factor = final_volume / initial_volume
    return {"dilution_factor": dilution_factor}

def calc_specific_activity_enzyme(enzyme_activity_units, total_protein_mass_mg):
    if enzyme_activity_units < 0 or total_protein_mass_mg <= 0: return {"error": "酶活性不能为负，总蛋白质量需为正。"}
    specific_activity = enzyme_activity_units / total_protein_mass_mg
    return {"specific_activity_units_per_mg": specific_activity}


# --- ALL_FORMULAS list combines existing from file and newly added ones ---
ALL_FORMULAS = [
    # === Physics Formulas ===
    {"name": "末速度 (v = u + at)", "subject": "物理", "func": calc_kinematics_v_uat, "inputs": [{"label": "初速度 u", "unit": "m/s", "key": "u"}, {"label": "加速度 a", "unit": "m/s²", "key": "a"}, {"label": "时间 t", "unit": "s", "key": "t"}], "outputs": [{"label": "末速度 v", "unit": "m/s", "key": "v"}], "formula_str": "v = u + a*t"},
    {"name": "位移 (s = ut + 0.5at²)", "subject": "物理", "func": calc_kinematics_s_ut_half_at2, "inputs": [{"label": "初速度 u", "unit": "m/s", "key": "u"}, {"label": "加速度 a", "unit": "m/s²", "key": "a"}, {"label": "时间 t", "unit": "s", "key": "t"}], "outputs": [{"label": "位移 s", "unit": "m", "key": "s"}], "formula_str": "s = ut + 0.5*a*t²"},
    {"name": "牛顿第二定律 (F = ma)", "subject": "物理", "func": calc_force_f_ma, "inputs": [{"label": "质量 m", "unit": "kg", "key": "m"}, {"label": "加速度 a", "unit": "m/s²", "key": "a"}], "outputs": [{"label": "力 F", "unit": "N", "key": "F"}], "formula_str": "F = m*a"},
    {"name": "动能 (KE = 0.5mv²)", "subject": "物理", "func": calc_energy_ke, "inputs": [{"label": "质量 m", "unit": "kg", "key": "m"}, {"label": "速度 v", "unit": "m/s", "key": "v"}], "outputs": [{"label": "动能 KE", "unit": "J", "key": "KE"}], "formula_str": "KE = 0.5*m*v²"},
    {"name": "重力势能 (PE = mgh)", "subject": "物理", "func": calc_energy_pe_mgh, "inputs": [{"label": "质量 m", "unit": "kg", "key": "m"}, {"label": "高度 h (相对参考点)", "unit": "m", "key": "h"}], "outputs": [{"label": "势能 PE", "unit": "J", "key": "PE"}], "formula_str": f"PE = mgh (g ≈ {GRAVITATIONAL_ACCELERATION:.3f} m/s²)"},
    {"name": "质能方程 (E = mc²)", "subject": "物理", "func": calc_einstein_emc2, "inputs": [{"label": "质量 m", "unit": "kg", "key": "m"}], "outputs": [{"label": "能量 E", "unit": "J", "key": "E"}], "formula_str": f"E = mc² (c ≈ {SPEED_OF_LIGHT:.0f} m/s)"},
    {"name": "欧姆定律 (计算 V)", "subject": "物理", "func": calc_ohms_law_voltage, "inputs": [{"label": "电流 I", "unit": "A", "key": "i"}, {"label": "电阻 R", "unit": "Ω", "key": "r"}], "outputs": [{"label": "电压 V", "unit": "V", "key": "V"}], "formula_str": "V = I*R"},
    {"name": "欧姆定律 (计算 I)", "subject": "物理", "func": calc_ohms_law_current, "inputs": [{"label": "电压 V", "unit": "V", "key": "v"}, {"label": "电阻 R", "unit": "Ω", "key": "r"}], "outputs": [{"label": "电流 I", "unit": "A", "key": "I"}], "formula_str": "I = V/R"},
    {"name": "欧姆定律 (计算 R)", "subject": "物理", "func": calc_ohms_law_resistance, "inputs": [{"label": "电压 V", "unit": "V", "key": "v"}, {"label": "电流 I", "unit": "A", "key": "i"}], "outputs": [{"label": "电阻 R", "unit": "Ω", "key": "R"}], "formula_str": "R = V/I"},
    {"name": "电功率 (P = IV)", "subject": "物理", "func": calc_power_iv, "inputs": [{"label": "电流 I", "unit": "A", "key": "i"}, {"label": "电压 V", "unit": "V", "key": "v"}], "outputs": [{"label": "功率 P", "unit": "W", "key": "P_watt"}], "formula_str": "P = I*V"},
    {"name": "电功率 (P = I²R)", "subject": "物理", "func": calc_power_i2r, "inputs": [{"label": "电流 I", "unit": "A", "key": "current_A"}, {"label": "电阻 R", "unit": "Ω", "key": "resistance_ohm"}], "outputs": [{"label": "功率 P", "unit": "W", "key": "P_power_watt"}], "formula_str": "P = I² * R"},
    {"name": "电功率 (P = V²/R)", "subject": "物理", "func": calc_power_v2r, "inputs": [{"label": "电压 V", "unit": "V", "key": "voltage_V"}, {"label": "电阻 R", "unit": "Ω", "key": "resistance_ohm"}], "outputs": [{"label": "功率 P", "unit": "W", "key": "P_power_watt"}], "formula_str": "P = V² / R"},
    {"name": "光子能量 (由频率 f)", "subject": "物理", "func": calc_photon_energy_freq, "inputs": [{"label": "频率 f", "unit": "Hz", "key": "f"}], "outputs": [{"label": "光子能量 E", "unit": "J", "key": "E_photon"}], "formula_str": f"E = hf (h ≈ {PLANCK_CONSTANT:.3e} J·s)"},
    {"name": "光子能量 (由波长 λ)", "subject": "物理", "func": calc_photon_energy_wavelength, "inputs": [{"label": "波长 λ", "unit": "m", "key": "lambda_val"}], "outputs": [{"label": "光子能量 E", "unit": "J", "key": "E_photon"}], "formula_str": "E = hc/λ"},
    {"name": "万有引力 (F = Gm₁m₂/r²)", "subject": "物理", "func": calc_universal_gravitation, "inputs": [{"label": "质量 m₁", "unit": "kg", "key": "m1"}, {"label": "质量 m₂", "unit": "kg", "key": "m2"}, {"label": "距离 r (质心间)", "unit": "m", "key": "r"}], "outputs": [{"label": "引力 F", "unit": "N", "key": "F_gravity"}], "formula_str": f"F = Gm₁m₂/r² (G ≈ {GRAVITATIONAL_CONSTANT_G:.3e})"},
    {"name": "单摆周期 (T = 2π√(L/g))", "subject": "物理", "func": calc_pendulum_period, "inputs": [{"label": "摆长 L", "unit": "m", "key": "L"}], "outputs": [{"label": "周期 T", "unit": "s", "key": "T_pendulum"}], "formula_str": "T = 2π√(L/g)"},
    {"name": "抛体运动 (h₀=0)", "subject": "物理", "func": lambda v0, angle_deg: calc_projectile_motion(v0, angle_deg, h0=0), "inputs": [{"label": "初速度 v₀", "unit": "m/s", "key": "v0"}, {"label": "发射角 θ (水平以上)", "unit": "度", "key": "angle_deg"}], "outputs": [{"label": "射程 R", "unit": "m", "key": "range"}, {"label": "最大高度 H_max", "unit": "m", "key": "max_height_abs"}, {"label": "飞行时间 t_flight", "unit": "s", "key": "time_of_flight"}], "formula_str": "斜抛 (发射高度h₀=0) 相关计算"},
    {"name": "抛体运动 (h₀≠0)", "subject": "物理", "func": calc_projectile_motion, "inputs": [{"label": "初速度 v₀", "unit": "m/s", "key": "v0"}, {"label": "发射角 θ (水平以上)", "unit": "度", "key": "angle_deg"}, {"label": "初始高度 h₀", "unit": "m", "key": "h0"}], "outputs": [{"label": "射程 R (落到y=0)", "unit": "m", "key": "range"}, {"label": "绝对最大高度 H_abs", "unit": "m", "key": "max_height_abs"}, {"label": "飞行时间 t_flight (落到y=0)", "unit": "s", "key": "time_of_flight"}], "formula_str": "斜抛 (发射高度h₀≠0) 相关计算"},
    {"name": "波速 (v = fλ)", "subject": "物理", "func": calc_wave_speed, "inputs": [{"label": "频率 f", "unit": "Hz", "key": "frequency"}, {"label": "波长 λ", "unit": "m", "key": "wavelength"}], "outputs": [{"label": "波速 v", "unit": "m/s", "key": "v_wave"}], "formula_str": "v = f * λ"},
    {"name": "比热容吸热 (Q = mcΔT)", "subject": "物理", "func": calc_specific_heat_energy, "inputs": [{"label": "质量 m", "unit": "kg", "key": "mass"}, {"label": "比热容 c", "unit": "J/(kg·K)", "key": "specific_heat_capacity"}, {"label": "温度变化 ΔT (T_final - T_initial)", "unit": "K or °C", "key": "delta_temp"}], "outputs": [{"label": "吸收/放出热量 Q", "unit": "J", "key": "Q_heat"}], "formula_str": "Q = mcΔT"},
    {"name": "库伦定律 (F = k|q₁q₂|/r²)", "subject": "物理", "func": calc_coulombs_law_force, "inputs": [{"label": "电荷量 q₁", "unit": "C", "key": "q1"}, {"label": "电荷量 q₂", "unit": "C", "key": "q2"}, {"label": "距离 r", "unit": "m", "key": "r_distance"}], "outputs": [{"label": "静电力 F (大小)", "unit": "N", "key": "F_electric"}], "formula_str": f"F = k|q₁q₂|/r² (k ≈ {K_COULOMB:.3e})"},
    {"name": "电容定义 (C = Q/V)", "subject": "物理", "func": calc_capacitance, "inputs": [{"label": "电荷量 Q", "unit": "C", "key": "Q_charge"}, {"label": "电势差 V", "unit": "V", "key": "V_potential_diff"}], "outputs": [{"label": "电容 C", "unit": "F (法拉)", "key": "C_capacitance"}], "formula_str": "C = Q/V"},
    {"name": "电容器储能 (U = ½CV²)", "subject": "物理", "func": calc_capacitor_energy, "inputs": [{"label": "电容 C", "unit": "F", "key": "C_capacitance"}, {"label": "电势差 V", "unit": "V", "key": "V_potential_diff"}], "outputs": [{"label": "储存能量 U", "unit": "J", "key": "U_capacitor"}], "formula_str": "U = 0.5 * C * V²"},
    {"name": "串联电阻 (R总 = R₁+R₂)", "subject": "物理", "func": calc_resistors_series_two, "inputs": [{"label": "电阻 R₁", "unit": "Ω", "key": "R1"}, {"label": "电阻 R₂", "unit": "Ω", "key": "R2"}], "outputs": [{"label": "总电阻 R_total", "unit": "Ω", "key": "R_total_series"}], "formula_str": "R_total = R₁ + R₂ (更多电阻继续相加)"},
    {"name": "并联电阻 (R总 = (R₁⁻¹+R₂⁻¹)⁻¹)", "subject": "物理", "func": calc_resistors_parallel_two, "inputs": [{"label": "电阻 R₁", "unit": "Ω", "key": "R1"}, {"label": "电阻 R₂", "unit": "Ω", "key": "R2"}], "outputs": [{"label": "总电阻 R_total", "unit": "Ω", "key": "R_total_parallel"}], "formula_str": "1/R_total = 1/R₁ + 1/R₂"},
    {"name": "斯涅尔定律 (计算 θ₂)", "subject": "物理", "func": calc_snells_law_theta2, "inputs": [{"label": "介质1折射率 n₁", "unit": "", "key": "n1"}, {"label": "入射角 θ₁ (0-90°)", "unit": "度", "key": "theta1_deg"}, {"label": "介质2折射率 n₂", "unit": "", "key": "n2"}], "outputs": [{"label": "折射角 θ₂ (信息)", "unit": "度/状态", "key": "theta2_info_content"}, {"label": "计算的sin(θ₂)", "unit": "", "key": "sin_theta2_calc_val"}, {"label": "折射角(度)", "unit":"°", "key":"theta2_deg_val"}], "formula_str": "n₁sin(θ₁) = n₂sin(θ₂)"},
    {"name": "透镜成像 (计算 di, M)", "subject": "物理", "func": calc_lens_formula_di, "inputs": [{"label": "焦距 f (+凸,-凹)", "unit": "m (或cm)", "key": "f_focal_length"}, {"label": "物距 do (实物为正)", "unit": "m (同f单位)", "key": "do_object_distance"}], "outputs": [{"label": "像距 di (实像为正)", "unit": "m (同f单位)", "key": "di_image_distance"}, {"label": "放大率 M (正立为正)", "unit": "", "key": "magnification_M"}], "formula_str": "1/f = 1/do + 1/di; M = -di/do"},
    {"name": "德布罗意波长 (λ = h/mv)", "subject": "物理", "func": calc_de_broglie_wavelength, "inputs": [{"label": "质量 m", "unit": "kg", "key": "mass_kg"}, {"label": "速度 v", "unit": "m/s", "key": "velocity_ms"}], "outputs": [{"label": "德布罗意波长 λ", "unit": "m", "key": "lambda_debroglie"}], "formula_str": "λ = h / (m*v) (h是普朗克常数)"},
    {"name": "多普勒效应 (声源远离,观察者静止)", "subject": "物理", "func": calc_doppler_sound_source_away_observer_still, "inputs": [{"label": "声源频率 f_src", "unit": "Hz", "key": "f_source"}, {"label": "声源速度 v_src (+远离)", "unit": "m/s", "key": "v_source_speed"}], "outputs": [{"label": "观测频率 f_obs", "unit": "Hz", "key": "f_observed_Hz"}], "formula_str": f"f_obs = f_src * (v_sound / (v_sound + v_src)) (v_sound ≈ {V_SOUND_AIR_STANDARD}m/s)"},
    {"name": "多普勒效应 (声源靠近,观察者静止)", "subject": "物理", "func": calc_doppler_sound_source_towards_observer_still, "inputs": [{"label": "声源频率 f_src", "unit": "Hz", "key": "f_source"}, {"label": "声源速度 v_src (+靠近)", "unit": "m/s", "key": "v_source_speed"}], "outputs": [{"label": "观测频率 f_obs", "unit": "Hz", "key": "f_observed_Hz"}], "formula_str": f"f_obs = f_src * (v_sound / (v_sound - v_src)) (v_sound ≈ {V_SOUND_AIR_STANDARD}m/s)"},
    {"name": "平均速度 (v = s/t)", "subject": "物理", "func": calc_average_velocity, "inputs": [{"label": "位移/路程 s", "unit": "m", "key": "distance"}, {"label": "时间 t", "unit": "s", "key": "time"}], "outputs": [{"label": "平均速度 v_avg", "unit": "m/s", "key": "v_avg"}], "formula_str": "v_avg = s / t"},
    {"name": "密度 (ρ = m/V)", "subject": "物理", "func": calc_density, "inputs": [{"label": "质量 m", "unit": "kg (或 g)", "key": "mass"}, {"label": "体积 V", "unit": "m³ (或 cm³/L)", "key": "volume"}], "outputs": [{"label": "密度 ρ", "unit": "kg/m³ (或 g/cm³)", "key": "density_rho"}], "formula_str": "ρ = m / V (注意单位一致性)"},
    {"name": "功 (W = Fdcosθ)", "subject": "物理", "func": calc_work_fd_cos_theta, "inputs": [{"label": "力 F", "unit": "N", "key": "force"}, {"label": "位移 d", "unit": "m", "key": "distance"}, {"label": "力与位移夹角 θ (可选,默认0)", "unit": "度", "key": "angle_deg"}], "outputs": [{"label": "功 W", "unit": "J", "key": "work_W"}], "formula_str": "W = F * d * cos(θ)"},
    {"name": "压强 (P = F/A)", "subject": "物理", "func": calc_pressure_FA, "inputs": [{"label": "力 F (垂直作用)", "unit": "N", "key": "force"}, {"label": "受力面积 A", "unit": "m²", "key": "area"}], "outputs": [{"label": "压强 P", "unit": "Pa (N/m²)", "key": "pressure_P"}], "formula_str": "P = F / A"},
    {"name": "胡克定律 (计算力 F = kx)", "subject": "物理", "func": calc_hookes_law_force, "inputs": [{"label": "弹簧劲度系数 k", "unit": "N/m", "key": "k_spring_constant"}, {"label": "形变量 x", "unit": "m", "key": "x_displacement"}], "outputs": [{"label": "弹力/外力 F (大小)", "unit": "N", "key": "force_spring_F"}], "formula_str": "F = k * x (x是形变量)"},
    {"name": "胡克定律 (计算形变量 x = F/k)", "subject": "物理", "func": calc_hookes_law_displacement, "inputs": [{"label": "作用力 F", "unit": "N", "key": "F_force_on_spring"}, {"label": "弹簧劲度系数 k", "unit": "N/m", "key": "k_spring_constant"}], "outputs": [{"label": "形变量 x (大小)", "unit": "m", "key": "x_displacement"}], "formula_str": "x = F / k"},
    {"name": "弹性势能 (PE_spring = ½kx²)", "subject": "物理", "func": calc_spring_potential_energy, "inputs": [{"label": "弹簧劲度系数 k", "unit": "N/m", "key": "k_spring_constant"}, {"label": "形变量 x", "unit": "m", "key": "x_displacement"}], "outputs": [{"label": "弹性势能 PE_spring", "unit": "J", "key": "PE_spring"}], "formula_str": "PE_spring = 0.5 * k * x²"},
    {"name": "向心力 (Fc = mv²/r)", "subject": "物理", "func": calc_centripetal_force, "inputs": [{"label": "质量 m", "unit": "kg", "key": "mass"}, {"label": "线速度 v", "unit": "m/s", "key": "velocity_tangential"}, {"label": "轨道半径 r", "unit": "m", "key": "radius_circular_path"}], "outputs": [{"label": "向心力 Fc", "unit": "N", "key": "Fc_centripetal_force"}], "formula_str": "Fc = m * v² / r"},
    {"name": "逃逸速度 (v_esc = √(2GM/R))", "subject": "物理", "func": calc_escape_velocity, "inputs": [{"label": "天体质量 M", "unit": "kg", "key": "M_celestial_body_mass"}, {"label": "天体半径 R", "unit": "m", "key": "R_celestial_body_radius"}], "outputs": [{"label": "逃逸速度 v_esc", "unit": "m/s", "key": "v_escape"}], "formula_str": f"v_esc = √(2GM/R) (G ≈ {GRAVITATIONAL_CONSTANT_G:.3e})"},
    {"name": "光子动量 (由波长 λ)", "subject": "物理", "func": calc_photon_momentum_from_wavelength, "inputs": [{"label": "波长 λ", "unit": "m", "key": "lambda_wavelength"}], "outputs": [{"label": "光子动量 p", "unit": "kg·m/s", "key": "p_photon_momentum"}], "formula_str": f"p = h/λ (h ≈ {PLANCK_CONSTANT:.3e})"},
    {"name": "光子动量 (由能量 E)", "subject": "物理", "func": calc_photon_momentum_from_energy, "inputs": [{"label": "光子能量 E", "unit": "J", "key": "E_photon_energy"}], "outputs": [{"label": "光子动量 p", "unit": "kg·m/s", "key": "p_photon_momentum"}], "formula_str": f"p = E/c (c ≈ {SPEED_OF_LIGHT:.0f})"},
    {"name": "角速度 (ω = v/r)", "subject": "物理", "func": calc_angular_velocity_v_r, "inputs": [{"label": "线速度 v", "unit": "m/s", "key": "v_linear_velocity"}, {"label": "半径 r", "unit": "m", "key": "r_radius"}], "outputs": [{"label": "角速度 ω", "unit": "rad/s", "key": "omega_angular_velocity"}], "formula_str": "ω = v / r"},
    {"name": "浮力 (F_buoyancy = ρ_fluid·V_displaced·g)", "subject": "物理", "func": calc_buoyancy_force, "inputs": [{"label": "流体密度 ρ_fluid", "unit": "kg/m³", "key": "fluid_density_kg_m3"}, {"label": "排开液体积 V_displaced", "unit": "m³", "key": "volume_displaced_m3"}], "outputs": [{"label": "浮力 F_buoyancy", "unit": "N", "key": "F_buoyancy_N"}], "formula_str": f"F_浮 = ρ_流体 * V_排 * g (g ≈ {GRAVITATIONAL_ACCELERATION:.3f} m/s²)"},
    {"name": "液体压强 (P = ρgh)", "subject": "物理", "func": calc_pressure_depth_rho_g_h, "inputs": [{"label": "流体密度 ρ", "unit": "kg/m³", "key": "fluid_density_kg_m3"}, {"label": "深度 h", "unit": "m", "key": "depth_h_m"}], "outputs": [{"label": "液体压强 P_fluid", "unit": "Pa", "key": "P_fluid_Pa"}], "formula_str": f"P = ρgh (g ≈ {GRAVITATIONAL_ACCELERATION:.3f} m/s²)"},
    {"name": "理想气体等压过程做功 (W = PΔV)", "subject": "物理", "func": calc_work_ideal_gas_isobaric, "inputs": [{"label": "恒定压强 P", "unit": "Pa", "key": "pressure_Pa"}, {"label": "初始体积 V₁", "unit": "m³", "key": "V_initial_m3"}, {"label": "最终体积 V₂", "unit": "m³", "key": "V_final_m3"}], "outputs": [{"label": "气体对外做功 W_by_gas", "unit": "J", "key": "W_by_gas_J"}, {"label": "体积变化 ΔV", "unit": "m³", "key": "delta_V_m3"}], "formula_str": "W_对外 = P * (V₂ - V₁)"},
    {"name": "卡诺热机效率 (η = 1 - Tc/Th)", "subject": "物理", "func": calc_thermal_efficiency_carnot, "inputs": [{"label": "高温热源温度 Th", "unit": "K", "key": "T_hot_K"}, {"label": "低温热源温度 Tc", "unit": "K", "key": "T_cold_K"}], "outputs": [{"label": "卡诺效率 η_carnot", "unit": "%", "key": "efficiency_carnot_percent"}], "formula_str": "η_carnot = 1 - (Tc / Th) (温度单位为开尔文 K)"},
    {"name": "洛伦兹力-磁场对运动电荷 (F = |q|vBsinθ)", "subject": "物理", "func": calc_magnetic_force_on_charge_qvBsin, "inputs": [{"label": "电荷量 q", "unit": "C", "key": "q_charge_C"}, {"label": "速度 v", "unit": "m/s", "key": "v_velocity_ms"}, {"label": "磁感应强度 B", "unit": "T (特斯拉)", "key": "B_field_T"}, {"label": "v与B夹角 θ (可选,默认90°)", "unit": "度", "key": "angle_deg_v_B"}], "outputs": [{"label": "磁场力 F_magnetic (大小)", "unit": "N", "key": "F_magnetic_N"}], "formula_str": "F = |q|vBsinθ"},
    {"name": "RL电路时间常数 (τ = L/R)", "subject": "物理", "func": calc_time_constant_RL_circuit, "inputs": [{"label": "电感 L", "unit": "H (亨利)", "key": "L_inductance_H"}, {"label": "电阻 R", "unit": "Ω", "key": "R_resistance_ohm"}], "outputs": [{"label": "时间常数 τ_RL", "unit": "s", "key": "tau_RL_s"}], "formula_str": "τ = L / R"},
    {"name": "RC电路时间常数 (τ = RC)", "subject": "物理", "func": calc_time_constant_RC_circuit, "inputs": [{"label": "电阻 R", "unit": "Ω", "key": "R_resistance_ohm"}, {"label": "电容 C", "unit": "F (法拉)", "key": "C_capacitance_F"}], "outputs": [{"label": "时间常数 τ_RC", "unit": "s", "key": "tau_RC_s"}], "formula_str": "τ = R * C"},
    {"name": "单缝衍射(暗纹位置)", "subject": "物理", "func": calc_single_slit_diffraction_minima, "inputs": [ {"label": "衍射级数 m (非零整数)", "unit": "", "key": "m_order"}, {"label": "波长 λ", "unit": "m", "key": "lambda_wavelength"}, {"label": "缝宽 a", "unit": "m", "key": "a_slit_width"}], "outputs": [{"label": "衍射角 θ (暗纹)", "unit": "度", "key": "theta_deg_minima"}, {"label": "sin(θ) 值", "unit": "", "key": "sin_theta_minima"}], "formula_str": "a sin(θ) = mλ (m=±1,±2,...)"},
    {"name": "双缝干涉(明纹位置)", "subject": "物理", "func": calc_double_slit_interference_maxima, "inputs": [ {"label": "干涉级数 m (整数)", "unit": "", "key": "m_order"}, {"label": "波长 λ", "unit": "m", "key": "lambda_wavelength"}, {"label": "缝间距 d", "unit": "m", "key": "d_slit_separation"}], "outputs": [{"label": "干涉角 θ (明纹)", "unit": "度", "key": "theta_deg_maxima"}, {"label": "sin(θ) 值", "unit": "", "key": "sin_theta_maxima"}], "formula_str": "d sin(θ) = mλ (m=0,±1,±2,...)"},
    {"name": "流体连续性方程 (v₂)", "subject": "物理", "func": calc_fluid_continuity, "inputs": [ {"label": "截面1面积 A₁", "unit": "m²", "key": "A1_area_m2"}, {"label": "截面1流速 v₁", "unit": "m/s", "key": "v1_velocity_ms"}, {"label": "截面2面积 A₂", "unit": "m²", "key": "A2_area_m2"}], "outputs": [{"label": "截面2流速 v₂", "unit": "m/s", "key": "v2_velocity_ms"}], "formula_str": "A₁v₁ = A₂v₂  => v₂ = (A₁v₁)/A₂"},
    {"name": "热力学第一定律 (ΔU)", "subject": "物理", "func": calc_first_law_thermo_deltaU, "inputs": [ {"label": "系统吸热 Q (+)", "unit": "J", "key": "Q_heat_J"}, {"label": "系统对外做功 W (+)", "unit": "J", "key": "W_work_done_by_system_J"}], "outputs": [{"label": "内能变化 ΔU", "unit": "J", "key": "delta_U_internal_energy_J"}], "formula_str": "ΔU = Q - W (W为系统对外做功)"},
    {"name": "RLC串联电路阻抗 (Z)", "subject": "物理", "func": calc_rlc_series_impedance, "inputs": [ {"label": "电阻 R", "unit": "Ω", "key": "R_ohm"}, {"label": "感抗 XL", "unit": "Ω", "key": "XL_ohm_inductive_reactance"}, {"label": "容抗 XC", "unit": "Ω", "key": "XC_ohm_capacitive_reactance"}], "outputs": [{"label": "总阻抗 Z", "unit": "Ω", "key": "Z_impedance_ohm"}], "formula_str": "Z = √[R² + (XL - XC)²]"},
    {"name": "理想变压器(电压)", "subject": "物理", "func": calc_transformer_voltage_turns, "inputs": [ {"label": "初级电压 Vp", "unit": "V", "key": "Vp_primary_V"}, {"label": "初级匝数 Np", "unit": "", "key": "Np_primary_turns"}, {"label": "次级匝数 Ns", "unit": "", "key": "Ns_secondary_turns"}], "outputs": [{"label": "次级电压 Vs", "unit": "V", "key": "Vs_secondary_V"}], "formula_str": "Vp/Vs = Np/Ns"},
    {"name": "斯特凡-玻尔兹曼定律(总辐射功率)", "subject": "物理", "func": calc_stefan_boltzmann_law_power, "inputs": [ {"label": "发射率 ε (0-1)", "unit": "", "key": "emissivity"}, {"label": "表面积 A", "unit": "m²", "key": "area_m2"}, {"label": "绝对温度 T", "unit": "K", "key": "T_kelvin"}], "outputs": [{"label": "辐射功率 P", "unit": "W", "key": "P_radiated_power_W"}], "formula_str": f"P = εσAT⁴ (σ ≈ {STEFAN_BOLTZMANN_CONSTANT_sigma:.3e} W/m²K⁴)"},
    {"name": "维恩位移定律(峰值波长)", "subject": "物理", "func": calc_wien_displacement_law_lambda_max, "inputs": [ {"label": "绝对温度 T", "unit": "K", "key": "T_kelvin"}], "outputs": [{"label": "峰值波长 λ_max", "unit": "m", "key": "lambda_max_m"}], "formula_str": f"λ_max * T = b (b ≈ {WIEN_DISPLACEMENT_CONSTANT_b:.3e} m·K)"},
    {"name": "LC振荡电路频率", "subject": "物理", "func": calc_lc_oscillation_frequency, "inputs": [ {"label": "电感 L", "unit": "H", "key": "L_inductance_H"}, {"label": "电容 C", "unit": "F", "key": "C_capacitance_F"}], "outputs": [{"label": "固有频率 f", "unit": "Hz", "key": "frequency_Hz_LC"}], "formula_str": "f = 1 / (2π√(LC))"},
    {"name": "圆孔衍射(艾里斑角半径)", "subject": "物理", "func": calc_round_aperture_diffraction_angle, "inputs": [ {"label": "波长 λ", "unit": "m", "key": "lambda_wavelength"}, {"label": "圆孔直径 D", "unit": "m", "key": "D_aperture_diameter"}], "outputs": [{"label": "艾里斑角半径 θ", "unit": "度", "key": "theta_deg_airy_radius"}, {"label": "sin(θ) 值", "unit": "", "key": "sin_theta_airy"}], "formula_str": "sin(θ) ≈ 1.22 * (λ/D) (第一暗环)"},
    {"name": "动摩擦力 (Fk = μkN)", "subject": "物理", "func": calc_kinetic_friction, "inputs": [{"label": "动摩擦因数 μk", "unit": "", "key": "mu_k"}, {"label": "正压力 N", "unit": "N", "key": "N_normal_force"}], "outputs": [{"label": "动摩擦力 Fk", "unit": "N", "key": "F_kinetic_friction_N"}], "formula_str": "Fk = μk * N"},
    {"name": "最大静摩擦力 (Fs_max = μsN)", "subject": "物理", "func": calc_static_friction_max, "inputs": [{"label": "静摩擦因数 μs", "unit": "", "key": "mu_s"}, {"label": "正压力 N", "unit": "N", "key": "N_normal_force"}], "outputs": [{"label": "最大静摩擦力 Fs_max", "unit": "N", "key": "F_static_friction_max_N"}], "formula_str": "Fs_max = μs * N"},
    {"name": "动量 (p = mv)", "subject": "物理", "func": calc_momentum, "inputs": [{"label": "质量 m", "unit": "kg", "key": "m_mass"}, {"label": "速度 v", "unit": "m/s", "key": "v_velocity"}], "outputs": [{"label": "动量 p", "unit": "kg·m/s", "key": "momentum_p"}], "formula_str": "p = m * v"},
    {"name": "冲量 (I = FΔt)", "subject": "物理", "func": calc_impulse, "inputs": [{"label": "力 F", "unit": "N", "key": "F_force"}, {"label": "时间变化量 Δt", "unit": "s", "key": "delta_t_time"}], "outputs": [{"label": "冲量 I", "unit": "N·s", "key": "impulse_I"}], "formula_str": "I = F * Δt"},
    {"name": "理想气体压强 (分子动理论)", "subject": "物理", "func": calc_pressure_ideal_gas_kinetic_theory, "inputs": [{"label": "粒子数 N", "unit": "", "key": "N_particles"}, {"label": "体积 V", "unit": "m³", "key": "V_volume"}, {"label": "平均平动动能 KE_avg", "unit": "J", "key": "KE_avg_particle"}], "outputs": [{"label": "压强 P", "unit": "Pa", "key": "pressure_Pa_kinetic"}], "formula_str": "P = (2/3) * (N/V) * KE_avg"},
    {"name": "理想气体方均根速率 (v_rms)", "subject": "物理", "func": calc_rms_velocity_ideal_gas, "inputs": [{"label": "摩尔质量 M", "unit": "kg/mol", "key": "M_molar_mass_kg_mol"}, {"label": "绝对温度 T", "unit": "K", "key": "T_kelvin"}], "outputs": [{"label": "方均根速率 v_rms", "unit": "m/s", "key": "v_rms_ms"}], "formula_str": "v_rms = √((3RT)/M)"},
    {"name": "点电荷电场强度 (E)", "subject": "物理", "func": calc_electric_field_point_charge, "inputs": [{"label": "点电荷 q", "unit": "C", "key": "q_charge_C"}, {"label": "距离 r", "unit": "m", "key": "r_distance_m"}], "outputs": [{"label": "电场强度 E", "unit": "N/C", "key": "E_field_N_C"}], "formula_str": "E = kq / r²"},
    {"name": "点电荷电势 (V)", "subject": "物理", "func": calc_electric_potential_point_charge, "inputs": [{"label": "点电荷 q", "unit": "C", "key": "q_charge_C"}, {"label": "距离 r", "unit": "m", "key": "r_distance_m"}], "outputs": [{"label": "电势 V", "unit": "V", "key": "V_potential_V"}], "formula_str": "V = kq / r"},
    {"name": "通电直导线磁场 (B)", "subject": "物理", "func": calc_magnetic_field_straight_wire, "inputs": [{"label": "电流 I", "unit": "A", "key": "I_current_A"}, {"label": "距离 r (到导线)", "unit": "m", "key": "r_distance_m"}], "outputs": [{"label": "磁场强度 B", "unit": "T", "key": "B_field_T"}], "formula_str": "B = (μ₀I) / (2πr)"},
    {"name": "法拉第电磁感应定律 (感应电动势)", "subject": "物理", "func": calc_induced_emf_faradays_law, "inputs": [{"label": "磁通量变化 ΔΦ", "unit": "Wb (韦伯)", "key": "delta_phi_magnetic_flux_Wb"}, {"label": "时间变化量 Δt", "unit": "s", "key": "delta_t_time_s"}], "outputs": [{"label": "感应电动势 emf", "unit": "V", "key": "emf_induced_V"}], "formula_str": "emf = - ΔΦ / Δt"},
    {"name": "电感器储能 (U = ½LI²)", "subject": "物理", "func": calc_energy_stored_inductor, "inputs": [{"label": "电感 L", "unit": "H", "key": "L_inductance_H"}, {"label": "电流 I", "unit": "A", "key": "I_current_A"}], "outputs": [{"label": "储存能量 U", "unit": "J", "key": "U_inductor_J"}], "formula_str": "U = 0.5 * L * I²"},
    {"name": "交流电路平均功率 (P_avg)", "subject": "物理", "func": calc_power_in_AC_circuit, "inputs": [{"label": "电压有效值 V_rms", "unit": "V", "key": "V_rms_V"}, {"label": "电流有效值 I_rms", "unit": "A", "key": "I_rms_A"}, {"label": "功率因数 cos(φ)", "unit": "", "key": "cos_phi_power_factor"}], "outputs": [{"label": "平均功率 P_avg", "unit": "W", "key": "P_avg_watt"}], "formula_str": "P_avg = V_rms * I_rms * cos(φ)"},
    {"name": "容抗 (XC)", "subject": "物理", "func": calc_capacitive_reactance, "inputs": [{"label": "频率 f", "unit": "Hz", "key": "f_frequency_Hz"}, {"label": "电容 C", "unit": "F", "key": "C_capacitance_F"}], "outputs": [{"label": "容抗 XC", "unit": "Ω", "key": "XC_ohm"}], "formula_str": "XC = 1 / (2πfC)"},
    {"name": "感抗 (XL)", "subject": "物理", "func": calc_inductive_reactance, "inputs": [{"label": "频率 f", "unit": "Hz", "key": "f_frequency_Hz"}, {"label": "电感 L", "unit": "H", "key": "L_inductance_H"}], "outputs": [{"label": "感抗 XL", "unit": "Ω", "key": "XL_ohm"}], "formula_str": "XL = 2πfL"},
    {"name": "RLC串联谐振频率", "subject": "物理", "func": calc_resonance_frequency_RLC, "inputs": [{"label": "电感 L", "unit": "H", "key": "L_inductance_H"}, {"label": "电容 C", "unit": "F", "key": "C_capacitance_F"}], "outputs": [{"label": "谐振频率 f_res", "unit": "Hz", "key": "f_resonance_Hz"}], "formula_str": "f_res = 1 / (2π√(LC))"},
    {"name": "光子能量 (由波长 λ, nm)", "subject": "物理", "func": calc_photon_energy_from_wavelength_nm, "inputs": [{"label": "波长 λ", "unit": "nm", "key": "lambda_nm"}], "outputs": [{"label": "光子能量 E", "unit": "J", "key": "E_photon_J"}, {"label": "光子能量 E", "unit": "eV", "key": "E_photon_eV"}], "formula_str": "E = hc/λ"},


    # === Chemistry Formulas ===
    {"name": "理想气体状态方程 (计算 P)", "subject": "化学", "func": calc_ideal_gas_pressure, "inputs": [{"label": "物质的量 n", "unit": "mol", "key": "n"}, {"label": "温度 T", "unit": "°C", "key": "temp_c"}, {"label": "体积 V", "unit": "L", "key": "vol_L"}], "outputs": [{"label": "压强 P", "unit": "Pa", "key": "P_Pa"}], "formula_str": f"P = nRT/V (R ≈ {R_GAS_CONSTANT:.3f} J/mol·K)"},
    {"name": "理想气体状态方程 (计算 V)", "subject": "化学", "func": calc_ideal_gas_volume, "inputs": [{"label": "物质的量 n", "unit": "mol", "key": "n_moles"}, {"label": "温度 T", "unit": "°C", "key": "temp_c"}, {"label": "压强 P", "unit": "Pa", "key": "pressure_Pa"}], "outputs": [{"label": "体积 V (升)", "unit": "L", "key": "V_liters"}, {"label": "体积 V (立方米)", "unit": "m³", "key": "V_cubic_meters"}], "formula_str": "V = nRT/P"},
    {"name": "理想气体状态方程 (计算 n)", "subject": "化学", "func": calc_ideal_gas_moles, "inputs": [{"label": "压强 P", "unit": "Pa", "key": "pressure_Pa"}, {"label": "体积 V", "unit": "L", "key": "vol_L"}, {"label": "温度 T", "unit": "°C", "key": "temp_c"}], "outputs": [{"label": "物质的量 n", "unit": "mol", "key": "n_moles_calculated"}], "formula_str": "n = PV/RT"},
    {"name": "理想气体状态方程 (计算 T)", "subject": "化学", "func": calc_ideal_gas_temp, "inputs": [{"label": "压强 P", "unit": "Pa", "key": "pressure_Pa"}, {"label": "体积 V", "unit": "L", "key": "vol_L"}, {"label": "物质的量 n", "unit": "mol", "key": "n_moles"}], "outputs": [{"label": "温度 T (开尔文)", "unit": "K", "key": "T_Kelvin"}, {"label": "温度 T (摄氏度)", "unit": "°C", "key": "T_Celsius"}], "formula_str": "T = PV/nR"},
    {"name": "物质的量 (n = mass/MolarMass)", "subject": "化学", "func": calc_moles_from_mass, "inputs": [{"label": "质量 mass", "unit": "g", "key": "mass_g"}, {"label": "摩尔质量 MolarMass", "unit": "g/mol", "key": "molar_mass_g_mol"}], "outputs": [{"label": "物质的量 n", "unit": "mol", "key": "n_moles"}], "formula_str": "n = mass / MolarMass"},
    {"name": "摩尔浓度 (M = n/V)", "subject": "化学", "func": calc_molarity_M, "inputs": [{"label": "溶质的物质的量 n", "unit": "mol", "key": "moles"}, {"label": "溶液体积 V", "unit": "L", "key": "vol_L"}], "outputs": [{"label": "摩尔浓度 M", "unit": "mol/L", "key": "M"}], "formula_str": "M = n_solute / V_solution_L"},
    {"name": "质量百分比 (%)", "subject": "化学", "func": calc_mass_percent, "inputs": [{"label": "溶质质量", "unit": "g", "key": "mass_solute_g"}, {"label": "溶液质量", "unit": "g", "key": "mass_solution_g"}], "outputs": [{"label": "质量百分比", "unit": "%", "key": "mass_percent"}], "formula_str": "(mass_solute / mass_solution) * 100%"},
    {"name": "质量摩尔浓度 (molality)", "subject": "化学", "func": calc_molality, "inputs": [{"label": "溶质物质的量 n_solute", "unit": "mol", "key": "moles_solute"}, {"label": "溶剂质量 m_solvent", "unit": "kg", "key": "mass_solvent_kg"}], "outputs": [{"label": "质量摩尔浓度 m", "unit": "mol/kg", "key": "m_molality"}], "formula_str": "molality = n_solute / m_solvent_kg"},
    {"name": "pH 计算 (pH = -log₁₀[H⁺])", "subject": "化学", "func": calc_ph_from_h, "inputs": [{"label": "H⁺ 浓度 [H⁺]", "unit": "mol/L", "key": "h_plus_conc"}], "outputs": [{"label": "pH", "unit": "", "key": "pH"}], "formula_str": "pH = -log₁₀[H⁺]"},
    {"name": "产率 (%)", "subject": "化学", "func": calc_percent_yield, "inputs": [{"label": "实际产量 (质量或摩尔)", "unit": "g (或mol)", "key": "actual_yield"}, {"label": "理论产量 (同上单位)", "unit": "g (或mol)", "key": "theoretical_yield"}], "outputs": [{"label": "产率", "unit": "%", "key": "percent_yield"}], "formula_str": "产率 = (实际产量 / 理论产量) * 100%"},
    {"name": "稀释定律 (M₁V₁=M₂V₂) - 计算 M₂", "subject": "化学", "func": calc_dilution_m2, "inputs": [{"label": "初始浓度 M₁", "unit": "mol/L", "key": "m1"}, {"label": "初始体积 V₁", "unit": "L (或其他)", "key": "v1"}, {"label": "最终体积 V₂", "unit": "L (与V₁同单位)", "key": "v2"}], "outputs": [{"label": "最终浓度 M₂", "unit": "mol/L", "key": "M2"}], "formula_str": "M₂ = (M₁V₁) / V₂"},
    {"name": "稀释定律 (M₁V₁=M₂V₂) - 计算 V₂", "subject": "化学", "func": calc_dilution_v2, "inputs": [{"label": "初始浓度 M₁", "unit": "mol/L", "key": "m1"}, {"label": "初始体积 V₁", "unit": "L (或其他)", "key": "v1"}, {"label": "最终浓度 M₂", "unit": "mol/L", "key": "m2"}], "outputs": [{"label": "所需最终体积 V₂", "unit": "L (与V₁同单位)", "key": "V2"}], "formula_str": "V₂ = (M₁V₁) / M₂"},
    {"name": "吉布斯自由能 (ΔG = ΔH - TΔS)", "subject": "化学", "func": calc_gibbs_free_energy, "inputs": [{"label": "焓变 ΔH", "unit": "kJ/mol", "key": "delta_H_kJ"}, {"label": "温度 T", "unit": "°C", "key": "temp_c"}, {"label": "熵变 ΔS", "unit": "J/(mol·K)", "key": "delta_S_J_K"}], "outputs": [{"label": "吉布斯自由能变 ΔG", "unit": "kJ/mol", "key": "delta_G_kJ"}], "formula_str": "ΔG = ΔH - TΔS (注意单位转换: ΔH→J, T→K)"},
    {"name": "一级反应半衰期 (t½ = ln(2)/k)", "subject": "化学", "func": calc_half_life_first_order, "inputs": [{"label": "速率常数 k", "unit": "s⁻¹ (或其它时间⁻¹)", "key": "k"}], "outputs": [{"label": "半衰期 t½", "unit": "s (与k时间单位一致)", "key": "t_half"}], "formula_str": "t½ = ln(2)/k ≈ 0.693/k"},
    {"name": "阿伦尼乌斯方程 (计算 k)", "subject": "化学", "func": calc_arrhenius_k, "inputs": [{"label": "指前因子 A", "unit": "(与k同单位)", "key": "A"}, {"label": "活化能 Ea", "unit": "kJ/mol", "key": "Ea_kJ_mol"}, {"label": "温度 T", "unit": "°C", "key": "temp_c"}], "outputs": [{"label": "速率常数 k", "unit": "(单位由A定)", "key": "k_rate_const"}], "formula_str": "k = Ae^(-Ea/RT) (Ea→J, T→K)"},
    {"name": "能斯特方程 (E_cell)", "subject": "化学", "func": calc_nernst_equation_Ecell, "inputs": [{"label": "标准电极电势 E°cell", "unit": "V", "key": "E_standard_cell_V"}, {"label": "转移电子数 n", "unit": "(整数)", "key": "n_electrons"}, {"label": "反应商 Q", "unit": "", "key": "Q_reaction_quotient"}, {"label": "温度 T (可选,默认25)", "unit": "°C", "key": "temp_c"}], "outputs": [{"label": "电池电动势 E_cell", "unit": "V", "key": "E_cell_V"}], "formula_str": "E_cell = E°_cell - (RT/nF)ln(Q)"},
    {"name": "亨-哈方程 (计算pH)", "subject": "化学", "func": calc_henderson_hasselbalch_ph, "inputs": [{"label": "弱酸的pKa", "unit": "", "key": "pKa"}, {"label": "共轭碱浓度 [A⁻]", "unit": "mol/L", "key": "A_minus_conc"}, {"label": "弱酸浓度 [HA]", "unit": "mol/L", "key": "HA_conc"}], "outputs": [{"label": "缓冲溶液pH", "unit": "", "key": "pH_buffer"}], "formula_str": "pH = pKa + log₁₀([A⁻]/[HA])"},
    {"name": "放射性衰变 (计算 Nt)", "subject": "化学", "func": calc_radioactive_decay_Nt, "inputs": [{"label": "初始量 N₀", "unit": "g, Bq, atoms etc.", "key": "N0_initial_amount"}, {"label": "经过时间 t", "unit": "s, days, years (同T½)", "key": "time_elapsed"}, {"label": "半衰期 T½", "unit": "s, days, years (同t)", "key": "half_life"}], "outputs": [{"label": "t 时剩余量 Nt", "unit": "(同N₀单位)", "key": "Nt_remaining"}], "formula_str": "Nt = N₀ * (0.5)^(t/T½)"},
    {"name": "STP气体摩尔数 (n=V/22.4)", "subject": "化学", "func": calc_gas_moles_at_STP, "inputs": [{"label": "STP下气体体积 V_STP", "unit": "L", "key": "volume_L_at_STP"}], "outputs": [{"label": "物质的量 n", "unit": "mol", "key": "n_moles_STP"}], "formula_str": f"n = V_L / {STP_MOLAR_VOLUME:.3f} L/mol (仅限STP: 0°C, 1 atm)"},
    {"name": "粒子数计算 (N = n·N_A)", "subject": "化学", "func": calc_number_of_particles, "inputs": [{"label": "物质的量 n", "unit": "mol", "key": "n_moles"}], "outputs": [{"label": "粒子数 N", "unit": "个", "key": "N_particles"}], "formula_str": f"N = n * N_A (N_A ≈ {AVOGADRO_CONSTANT:.3e} mol⁻¹)"},
    {"name": "摩尔溶解度 s (由Ksp, AB型)", "subject": "化学", "func": calc_molar_solubility_s_from_Ksp_AB, "inputs": [{"label": "溶度积 Ksp (AB型)", "unit": "", "key": "Ksp_AB"}], "outputs": [{"label": "摩尔溶解度 s", "unit": "mol/L", "key": "s_molar_solubility"}], "formula_str": "s = √Ksp (适用于 MX 型微溶盐, 如 AgCl)"},
    {"name": "摩尔溶解度 s (由Ksp, A₂B或AB₂型)", "subject": "化学", "func": calc_molar_solubility_s_from_Ksp_A2B_or_AB2, "inputs": [{"label": "溶度积 Ksp (A₂B或AB₂型)", "unit": "", "key": "Ksp_A2B_or_AB2"}], "outputs": [{"label": "摩尔溶解度 s", "unit": "mol/L", "key": "s_molar_solubility"}], "formula_str": "s = (Ksp/4)^(1/3) (适用于 M₂X 或 MX₂ 型, 如 Ag₂CrO₄, Mg(OH)₂)"},
    {"name": "pH与pOH换算 (25°C): 计算pH", "subject": "化学", "func": calc_pH_from_pOH, "inputs": [{"label": "pOH 值", "unit": "", "key": "pOH"}, {"label": "温度 T(可选,当前仅支持25)", "unit": "°C", "key": "temp_c"}], "outputs": [{"label": "计算的 pH 值", "unit": "", "key": "pH_calculated"}], "formula_str": "pH = 14 - pOH (25°C时 pKw=14)"},
    {"name": "pH与pOH换算 (25°C): 计算pOH", "subject": "化学", "func": calc_pOH_from_pH, "inputs": [{"label": "pH 值", "unit": "", "key": "pH"}, {"label": "温度 T(可选,当前仅支持25)", "unit": "°C", "key": "temp_c"}], "outputs": [{"label": "计算的 pOH 值", "unit": "", "key": "pOH_calculated"}], "formula_str": "pOH = 14 - pH (25°C时 pKw=14)"},
    {"name": "依数性-温度变化 (ΔT=iKm)", "subject": "化学", "func": calc_colligative_temp_change, "inputs": [{"label": "范特霍夫因子 i (≥1)", "unit": "", "key": "i_vant_hoff"}, {"label": "摩尔常数 K (Kb或Kf)", "unit": "°C·kg/mol", "key": "K_molal_constant"}, {"label": "质量摩尔浓度 m", "unit": "mol/kg", "key": "m_molality"}], "outputs": [{"label": "沸点升高/凝固点降低 ΔT", "unit": "°C", "key": "delta_T_change"}], "formula_str": "ΔT = i * K * m (Kb水≈0.512, Kf水≈1.86)"},
    {"name": "溶液凝固点 (Tf_sol)", "subject": "化学", "func": calc_solution_freezing_point, "inputs": [ {"label": "纯溶剂凝固点Tf°(可选,默认水)", "unit": "°C", "key": "Tf_pure_solvent_C"},{"label": "凝固点降低值 ΔTf", "unit": "°C", "key": "delta_Tf_depression"} ], "outputs": [{"label": "溶液凝固点 Tf_sol", "unit": "°C", "key": "Tf_solution_C"}], "formula_str": "Tf_solution = Tf°_solvent - ΔTf (ΔTf是正值)"},
    {"name": "溶液沸点 (Tb_sol)", "subject": "化学", "func": calc_solution_boiling_point, "inputs": [ {"label": "纯溶剂沸点Tb°(可选,默认水)", "unit": "°C", "key": "Tb_pure_solvent_C"}, {"label": "沸点升高值 ΔTb", "unit": "°C", "key": "delta_Tb_elevation"}], "outputs": [{"label": "溶液沸点 Tb_sol", "unit": "°C", "key": "Tb_solution_C"}], "formula_str": "Tb_solution = Tb°_solvent + ΔTb (ΔTb是正值)"},
    {"name": "拉乌尔定律-蒸气压降低", "subject": "化学", "func": calc_raoults_law_vapor_pressure_solution, "inputs": [ {"label": "纯溶剂蒸气压 P°", "unit": "atm (或其他)", "key": "P0_solvent_atm"}, {"label": "溶剂的摩尔分数 X_溶剂", "unit": "(0-1)", "key": "X_solvent_mole_fraction"}], "outputs": [{"label": "溶液蒸气压 P_溶液", "unit": "(同P°单位)", "key": "P_solution_atm"}], "formula_str": "P_溶液 = X_溶剂 * P°_溶剂 (理想溶液)"},
    {"name": "弱酸解离度 α (近似/精确)", "subject": "化学", "func": calc_degree_of_dissociation_weak_acid, "inputs": [ {"label": "酸的电离常数 Ka", "unit": "", "key": "Ka_ionization_const"}, {"label": "弱酸初始浓度 C₀", "unit": "mol/L", "key": "C0_initial_conc_M"}], "outputs": [{"label": "解离度 α (分数)", "unit": "", "key": "alpha_dissociation_fraction"}, {"label": "解离百分比 (%)", "unit": "%", "key": "alpha_percent"}], "formula_str": "Ka = α²C₀/(1-α) 或 α ≈ √(Ka/C₀)"},
    {"name": "水的离子积 Kw(T) (特定温度)", "subject": "化学", "func": calc_ionic_product_water_Kw_approx_temp, "inputs": [ {"label": "温度 T", "unit": "°C", "key": "temp_c"}], "outputs": [{"label": "水的离子积 Kw", "unit": "", "key": "Kw_ionic_product"}, {"label": "pKw", "unit": "", "key": "pKw"}, {"label": "使用温度 T", "unit": "°C", "key": "temp_used_C"}], "formula_str": "Kw = [H⁺][OH⁻] (随温度变化,提供常见值)"},
    {"name": "零级反应半衰期 (t½)", "subject": "化学", "func": calc_half_life_zero_order, "inputs": [{"label": "初始浓度 [A]₀", "unit": "mol/L (或其他)", "key": "A0_initial_conc_M"}, {"label": "速率常数 k", "unit": "浓度单位/时间单位", "key": "k_rate_constant_M_s"}], "outputs": [{"label": "零级反应半衰期 t½", "unit": "时间单位 (同k)", "key": "t_half_zero_order_s"}], "formula_str": "t½ = [A]₀ / (2k)"},
    {"name": "二级反应半衰期 (t½)", "subject": "化学", "func": calc_half_life_second_order, "inputs": [{"label": "初始浓度 [A]₀", "unit": "mol/L", "key": "A0_initial_conc_M"}, {"label": "速率常数 k", "unit": "(mol/L)⁻¹·s⁻¹ (或浓度⁻¹·时间⁻¹)", "key": "k_rate_constant_M_inv_s_inv"}], "outputs": [{"label": "二级反应半衰期 t½", "unit": "s (或其他时间)", "key": "t_half_second_order_s"}], "formula_str": "t½ = 1 / (k * [A]₀)"},
    {"name": "法拉第电解定律 (计算产物质量)", "subject": "化学", "func": calc_faradays_law_mass_electrolysis, "inputs": [{"label": "电流 I", "unit": "A", "key": "current_A"}, {"label": "时间 t", "unit": "s", "key": "time_s"}, {"label": "产物摩尔质量 M", "unit": "g/mol", "key": "M_molar_mass_g_mol"}, {"label": "转移电子数 n (每摩尔产物)", "unit": "(整数)", "key": "n_electrons_transferred"}], "outputs": [{"label": "析出/产生质量 m", "unit": "g", "key": "mass_deposited_g"}, {"label": "产生摩尔数", "unit": "mol", "key": "moles_produced"}], "formula_str": f"m = (I * t * M) / (n * F) (F ≈ {FARADAY_CONSTANT_DERIVED:.0f} C/mol)"},
    {"name": "渗透压 (Π = iMRT)", "subject": "化学", "func": calc_osmotic_pressure_MRT, "inputs": [{"label": "摩尔浓度 M (溶质总)", "unit": "mol/L", "key": "M_molarity_mol_L"}, {"label": "绝对温度 T", "unit": "K", "key": "temp_K"}, {"label": "范特霍夫因子 i (可选,默认1)", "unit": "", "key": "i_vant_hoff"}, {"label": "气体常数 R(L·atm/mol·K)(可选,默认0.0821)", "unit": "0.0821", "key": "R_gas_const_L_atm_mol_K"}], "outputs": [{"label": "渗透压 Π", "unit": "atm", "key": "osmotic_pressure_atm"}, {"label": "渗透压 Π", "unit": "Pa", "key": "osmotic_pressure_Pa"}], "formula_str": "Π = iMRT"},
    {"name": "亨利定律(气体溶解度)", "subject": "化学", "func": calc_henrys_law_solubility, "inputs": [ {"label": "亨利常数 kH", "unit": "mol/(L·atm) or M/atm", "key": "kH_henry_const_M_atm"}, {"label": "气体分压 P_gas", "unit": "atm", "key": "P_partial_pressure_gas_atm"}], "outputs": [{"label": "气体溶解度 S", "unit": "mol/L (M)", "key": "S_solubility_M"}], "formula_str": "S = kH * P_gas"},
    {"name": "平衡常数 Kc 与 Kp 转换", "subject": "化学", "func": calc_Kp_from_Kc, "inputs": [ {"label": "浓度平衡常数 Kc", "unit": "", "key": "Kc_equilibrium_const_conc"}, {"label": "绝对温度 T", "unit": "K", "key": "T_kelvin"}, {"label": "气体摩尔数变化 Δn_gas", "unit": "", "key": "delta_n_moles_gas"}, {"label": "气体常数R (L·atm/mol·K)(可选,默认0.0821)", "unit": "0.0821", "key":"R_gas_const_L_atm_mol_K"}], "outputs": [{"label": "压力平衡常数 Kp", "unit": "", "key": "Kp_equilibrium_const_pressure"}], "formula_str": "Kp = Kc * (RT)^(Δn_gas)"},
    {"name": "玻尔模型能级跃迁(类氢原子Z=1)", "subject": "化学", "func": calc_bohr_model_energy_transition, "inputs": [ {"label": "初始能级 ni (正整数)", "unit": "", "key": "ni_initial_level"}, {"label": "最终能级 nf (正整数)", "unit": "", "key": "nf_final_level"}], "outputs": [{"label": "能量变化 ΔE", "unit": "J", "key": "delta_E_J"}, {"label": "光子能量 |ΔE|", "unit": "J", "key": "photon_energy_J_abs"}, {"label": "光子波长 λ", "unit": "m", "key": "photon_wavelength_m"}, {"label": "光子频率 f", "unit": "Hz", "key": "photon_frequency_Hz"}, {"label": "跃迁类型", "unit": "", "key": "transition_type"}], "formula_str": f"ΔE = -RH * (1/nf² - 1/ni²) (RH≈2.18e-18 J)"},
    {"name": "反应热 (由生成焓计算)", "subject": "化学", "func": calc_heat_of_reaction_from_enthalpies_of_formation, "inputs": [
        {"label": "化学计量数 (字典格式: {'产物A': 1, '反应物B': -1})", "unit": "", "key": "stoich_coeffs"},
        {"label": "产物生成焓 (字典格式: {'产物A': -393.5})", "unit": "kJ/mol", "key": "delta_Hf_products_kJ_mol"},
        {"label": "反应物生成焓 (字典格式: {'反应物B': -241.8})", "unit": "kJ/mol", "key": "delta_Hf_reactants_kJ_mol"}
        ], "outputs": [{"label": "反应热 ΔH_rxn", "unit": "kJ/mol", "key": "delta_H_rxn_kJ_mol"}], "formula_str": "ΔH_rxn = Σ(ν_p * ΔHf_p) - Σ(ν_r * ΔHf_r)"},
    {"name": "平衡常数 K (由 ΔG° 计算)", "subject": "化学", "func": calc_equilibrium_constant_from_gibbs_free_energy, "inputs": [
        {"label": "标准吉布斯自由能变 ΔG°", "unit": "kJ/mol", "key": "delta_G_standard_kJ_mol"},
        {"label": "温度 T", "unit": "°C", "key": "temp_c"}
        ], "outputs": [{"label": "平衡常数 K", "unit": "", "key": "K_equilibrium_constant"}], "formula_str": "ΔG° = -RT ln K"},
    {"name": "范德华方程 (计算压强 P)", "subject": "化学", "func": calc_van_der_waals_pressure, "inputs": [
        {"label": "物质的量 n", "unit": "mol", "key": "n_moles"},
        {"label": "体积 V", "unit": "L", "key": "V_volume_L"},
        {"label": "温度 T", "unit": "°C", "key": "T_celsius"},
        {"label": "范德华常数 a", "unit": "Pa·m⁶/mol²", "key": "a_vdw"},
        {"label": "范德华常数 b", "unit": "m³/mol", "key": "b_vdw"}
        ], "outputs": [{"label": "压强 P_vdw", "unit": "Pa", "key": "P_van_der_waals_Pa"}], "formula_str": "(P + a(n/V)²)(V/n - b) = RT"},
    {"name": "定容热容 (Cv)", "subject": "化学", "func": calc_heat_capacity_constant_volume, "inputs": [
        {"label": "内能变化 ΔU", "unit": "J", "key": "delta_U_J"},
        {"label": "温度变化 ΔT", "unit": "K", "key": "delta_T_K"}
        ], "outputs": [{"label": "定容热容 Cv", "unit": "J/K", "key": "Cv_J_K"}], "formula_str": "Cv = ΔU / ΔT"},
    {"name": "定压热容 (Cp)", "subject": "化学", "func": calc_heat_capacity_constant_pressure, "inputs": [
        {"label": "焓变 ΔH", "unit": "J", "key": "delta_H_J"},
        {"label": "温度变化 ΔT", "unit": "K", "key": "delta_T_K"}
        ], "outputs": [{"label": "定压热容 Cp", "unit": "J/K", "key": "Cp_J_K"}], "formula_str": "Cp = ΔH / ΔT"},


    # === Biology Formulas ===
    {"name": "指数种群增长 (Nt = N₀e^(rt))", "subject": "生物", "func": calc_population_growth_exponential, "inputs": [{"label": "初始种群 N₀", "unit": "个体", "key": "N0"}, {"label": "瞬时增长率 r (可正可负)", "unit": "(例如 0.05)", "key": "r_rate"}, {"label": "时间 t", "unit": "(年/月等,与r单位一致)", "key": "t_time"}], "outputs": [{"label": "t 时种群 Nt", "unit": "个体", "key": "Nt"}], "formula_str": "Nt = N₀ * e^(r*t)"},
    {"name": "哈迪-温伯格 (由aa频率 q²算)", "subject": "生物", "func": calc_hardy_weinberg_from_q2, "inputs": [{"label": "隐性纯合子频率 q² (aa)", "unit": "(0-1)", "key": "q_squared"}], "outputs": [{"label": "p (A 等位基因频率)", "unit": "", "key": "p_A_freq"}, {"label": "q (a 等位基因频率)", "unit": "", "key": "q_a_freq"}, {"label": "p² (AA 基因型频率)", "unit": "", "key": "p_squared_AA"}, {"label": "2pq (Aa 基因型频率)", "unit": "", "key": "two_pq_Aa"}, {"label": "q² (aa 基因型频率 - 输入值)", "unit": "", "key": "q_squared_aa_input"}], "formula_str": "由隐性纯合子(aa)的频率(q²)推算：q=√q², p=1-q, 及基因型频率p², 2pq."},
    {"name": "显微镜放大倍数", "subject": "生物", "func": calc_magnification, "inputs": [{"label": "图像大小", "unit": "mm (或其他)", "key": "image_size"}, {"label": "实际大小", "unit": "mm (同上)", "key": "actual_size"}], "outputs": [{"label": "放大倍数", "unit": "X", "key": "magnification"}], "formula_str": "总放大倍数 = 图像观察大小 / 物体实际大小"},
    {"name": "BMI (身体质量指数)", "subject": "生物", "func": calc_bmi, "inputs": [{"label": "体重", "unit": "kg", "key": "weight_kg"}, {"label": "身高", "unit": "cm", "key": "height_cm"}], "outputs": [{"label": "BMI 指数", "unit": "", "key": "BMI"}, {"label": "分类 (中国标准)", "unit": "", "key": "Category"}], "formula_str": "BMI = 体重(kg) / (身高(m))²"},
    {"name": "逻辑斯谛增长 (N(t))", "subject": "生物", "func": calc_logistic_growth_Nt, "inputs": [{"label": "初始种群 N₀", "unit": "个体", "key": "N0"}, {"label": "环境容纳量 K", "unit": "个体", "key": "K_capacity"}, {"label": "内禀增长率 r", "unit": "(e.g. 0.05)", "key": "r_rate"}, {"label": "时间 t", "unit": "(单位同r)", "key": "t_time"}], "outputs": [{"label": "t 时种群 N(t)", "unit": "个体", "key": "Nt_logistic"}], "formula_str": "N(t) = K / (1 + ((K-N₀)/N₀) * e^(-rt))"},
    {"name": "米氏方程 (计算反应速率 V)", "subject": "生物", "func": calc_michaelis_menten_V, "inputs": [{"label": "最大反应速率 Vmax", "unit": "浓度/时间", "key": "Vmax"}, {"label": "底物浓度 [S]", "unit": "浓度 (同Km)", "key": "S_conc"}, {"label": "米氏常数 Km", "unit": "浓度 (同S)", "key": "Km"}], "outputs": [{"label": "反应速率 V", "unit": "(同Vmax单位)", "key": "V_reaction_rate"}], "formula_str": "V = (Vmax * [S]) / (Km + [S])"},
    {"name": "标记重捕法 (估算种群 N)", "subject": "生物", "func": calc_mark_recapture_N, "inputs": [{"label": "初次标记数 M", "unit": "个体", "key": "M_marked_first"}, {"label": "二次捕捉总数 C (含已标记和未标记)", "unit": "个体", "key": "C_captured_second"}, {"label": "二次捕捉中已标记数 R", "unit": "个体", "key": "R_recaptured_marked"}], "outputs": [{"label": "种群数量估计 N", "unit": "个体", "key": "N_population_estimate"}], "formula_str": "N = (M * C) / R (林肯-彼得森指数)"},
    {"name": "光合/呼吸速率 (气体法, STP)", "subject": "生物", "func": calc_photosynthesis_respiration_rate, "inputs": [{"label": "气体体积变化量 ΔV_gas", "unit": "mL", "key": "gas_change_volume_mL"}, {"label": "时间 Δt", "unit": "min", "key": "time_min"}, {"label": "生物量 (干重/鲜重)", "unit": "g", "key": "biomass_g"}, {"label": "气体摩尔质量 M_gas (可选,默认O₂)", "unit": "g/mol", "key": "molar_mass_gas_g_mol"}], "outputs": [{"label": "速率", "unit": "µmol_gas/(g·min)", "key": "rate_umol_g_min"}, {"label": "气体摩尔数变化", "unit": "mol", "key": "moles_gas_evolved_consumed"}], "formula_str": "速率 = (ΔV_gas_STP / Vm_STP) * 10⁶ / (biomass * Δt)"},
    {"name": "Lineweaver-Burk 作图点 (1/V, 1/[S])", "subject": "生物", "func": calc_lineweaver_burk_params, "inputs": [{"label": "底物浓度 [S]", "unit": "M (或其他浓度)", "key": "S_substrate_conc"}, {"label": "初反应速率 V", "unit": "浓度/时间 (同Vmax)", "key": "V_initial_velocity"}], "outputs": [{"label": "1/[S]", "unit": "M⁻¹ (或浓度⁻¹)", "key": "inv_S"}, {"label": "1/V", "unit": "(浓度/时间)⁻¹", "key": "inv_V"}], "formula_str": "1/V = (Km/Vmax)(1/[S]) + 1/Vmax (计算单点互易值)"},
    {"name": "基础代谢率BMR (Mifflin-St Jeor)", "subject": "生物", "func": calc_bmr_harris_benedict, "inputs": [{"label": "体重 W", "unit": "kg", "key": "weight_kg"}, {"label": "身高 H", "unit": "cm", "key": "height_cm"}, {"label": "年龄 A", "unit": "岁", "key": "age_years"}, {"label": "男性?(True/1或False/0)", "unit": "布尔值(1或0)", "key": "gender_male_bool"}], "outputs": [{"label": "BMR (Mifflin-St Jeor)", "unit": "kcal/day", "key": "bmr_kcal_day_MifflinStJeor"}], "formula_str": "男:10W+6.25H-5A+5; 女:10W+6.25H-5A-161"},
    {"name": "心输出量 (CO)", "subject": "生物", "func": calc_cardiac_output, "inputs": [ {"label": "每搏输出量 SV", "unit": "mL/beat", "key": "stroke_volume_mL_beat"}, {"label": "心率 HR", "unit": "beats/min", "key": "heart_rate_bpm"}], "outputs": [{"label": "心输出量 CO", "unit": "L/min", "key": "CO_L_min"}], "formula_str": "CO = SV (mL/beat) * HR (beats/min) / 1000"},
    {"name": "呼吸商 (RQ)", "subject": "生物", "func": calc_respiratory_quotient, "inputs": [ {"label": "CO₂ 产生摩尔数", "unit": "mol", "key": "CO2_eliminated_moles"}, {"label": "O₂ 消耗摩尔数", "unit": "mol", "key": "O2_consumed_moles"}], "outputs": [{"label": "呼吸商 RQ", "unit": "", "key": "RQ"}], "formula_str": "RQ = CO₂产生量 / O₂消耗量 (摩尔比)"},
    {"name": "估算最大心率 (Tanaka)", "subject": "生物", "func": calc_max_heart_rate_estimate_tanaka, "inputs": [ {"label": "年龄", "unit": "岁", "key": "age_years"}], "outputs": [{"label": "估算最大心率 (Tanaka法)", "unit": "bpm", "key": "hr_max_bpm_tanaka"}], "formula_str": "HRmax = 208 – (0.7 × 年龄)"},
    {"name": "渗透势 ψs (理想溶液)", "subject": "生物", "func": calc_water_potential_osmotic, "inputs": [ {"label": "溶质摩尔浓度 C", "unit": "mol/L", "key": "C_molar_conc_mol_L"}, {"label": "绝对温度 T", "unit": "K", "key": "T_kelvin"}, {"label": "范特霍夫因子 i (可选,默认1)", "unit": "", "key": "i_vant_hoff"}, {"label": "气体常数 R (可选,默认MPa适用)", "unit":"MPa·L/mol·K", "key":"R_gas_const_MPa_L_mol_K"}], "outputs": [{"label": "渗透势 ψs", "unit": "MPa", "key": "psi_s_osmotic_potential_MPa"}], "formula_str": "ψs = -iCRT (R = 0.008314 MPa·L/mol·K)"},
    {"name": "倍增时间 (由增长率r)", "subject": "生物", "func": calc_doubling_time_from_growth_rate, "inputs": [ {"label": "比增长速率 r", "unit": "时间⁻¹ (如 h⁻¹, day⁻¹)", "key": "r_specific_growth_rate_per_time"}], "outputs": [{"label": "倍增时间 t_double", "unit": "时间 (同r单位)", "key": "t_doubling"}], "formula_str": "t_double = ln(2) / r ≈ 0.693 / r"},
    {"name": "竞争性抑制表观Km (Km_app)", "subject": "生物", "func": calc_competitive_inhibition_apparent_Km, "inputs": [ {"label": "原始 Km", "unit": "浓度单位 (M, μM等)", "key": "Km_original_M"}, {"label": "抑制剂浓度 [I]", "unit": "(同Km和Ki单位)", "key": "I_inhibitor_conc_M"}, {"label": "抑制常数 Ki", "unit": "(同Km和[I]单位)", "key": "Ki_inhibitor_const_M"}], "outputs": [{"label": "表观 Km (Km_app)", "unit": "(同Km单位)", "key": "Km_apparent_M"}], "formula_str": "Km_app = Km * (1 + [I]/Ki)"},
    {"name": "种群密度", "subject": "生物", "func": calc_population_density, "inputs": [{"label": "种群大小", "unit": "个体", "key": "population_size"}, {"label": "面积", "unit": "m² (或其他)", "key": "area"}], "outputs": [{"label": "种群密度", "unit": "个体/m² (或其他)", "key": "population_density"}], "formula_str": "密度 = 种群大小 / 面积"},
    {"name": "出生率 (per capita)", "subject": "生物", "func": calc_birth_rate, "inputs": [{"label": "出生数", "unit": "个体", "key": "births"}, {"label": "种群大小", "unit": "个体", "key": "population_size"}, {"label": "时间单位 (可选,默认1)", "unit": "年/月/天等", "key": "time_unit"}], "outputs": [{"label": "出生率", "unit": "per capita per time", "key": "birth_rate_per_capita_per_time"}], "formula_str": "出生率 = (出生数 / 种群大小) / 时间"},
    {"name": "死亡率 (per capita)", "subject": "生物", "func": calc_death_rate, "inputs": [{"label": "死亡数", "unit": "个体", "key": "deaths"}, {"label": "种群大小", "unit": "个体", "key": "population_size"}, {"label": "时间单位 (可选,默认1)", "unit": "年/月/天等", "key": "time_unit"}], "outputs": [{"label": "死亡率", "unit": "per capita per time", "key": "death_rate_per_capita_per_time"}], "formula_str": "死亡率 = (死亡数 / 种群大小) / 时间"},
    {"name": "种群增长率 (出生-死亡)", "subject": "生物", "func": calc_population_growth_rate_birth_death, "inputs": [{"label": "出生率 (per capita)", "unit": "时间⁻¹", "key": "birth_rate"}, {"label": "死亡率 (per capita)", "unit": "时间⁻¹", "key": "death_rate"}, {"label": "种群大小", "unit": "个体", "key": "population_size"}], "outputs": [{"label": "种群增长率", "unit": "个体/时间", "key": "population_growth_rate"}], "formula_str": "增长率 = (出生率 - 死亡率) * 种群大小"},
    {"name": "等位基因频率", "subject": "生物", "func": calc_allele_frequency, "inputs": [{"label": "等位基因计数", "unit": "", "key": "count_of_allele"}, {"label": "总等位基因数", "unit": "", "key": "total_alleles"}], "outputs": [{"label": "等位基因频率", "unit": "", "key": "allele_frequency"}], "formula_str": "频率 = 等位基因计数 / 总等位基因数"},
    {"name": "基因型频率", "subject": "生物", "func": calc_genotype_frequency, "inputs": [{"label": "基因型计数", "unit": "", "key": "count_of_genotype"}, {"label": "总个体数", "unit": "", "key": "total_individuals"}], "outputs": [{"label": "基因型频率", "unit": "", "key": "genotype_frequency"}], "formula_str": "频率 = 基因型计数 / 总个体数"},
    {"name": "酶催化效率 (kcat/Km)", "subject": "生物", "func": calc_enzyme_efficiency_kcat_Km, "inputs": [{"label": "催化常数 kcat", "unit": "时间⁻¹", "key": "kcat"}, {"label": "米氏常数 Km", "unit": "浓度", "key": "Km"}], "outputs": [{"label": "酶催化效率", "unit": "浓度⁻¹·时间⁻¹", "key": "enzyme_efficiency_kcat_per_Km"}], "formula_str": "效率 = kcat / Km"},
    {"name": "哈迪-温伯格 (由p,q算)", "subject": "生物", "func": calc_hardy_weinberg_from_p_q, "inputs": [{"label": "等位基因频率 p (如 A)", "unit": "(0-1)", "key": "p_freq"}, {"label": "等位基因频率 q (如 a)", "unit": "(0-1)", "key": "q_freq"}], "outputs": [{"label": "p (输入值)", "unit": "", "key": "p_freq_input"}, {"label": "q (输入值)", "unit": "", "key": "q_freq_input"}, {"label": "p² (AA 基因型频率)", "unit": "", "key": "p_squared_AA"}, {"label": "q² (aa 基因型频率)", "unit": "", "key": "q_squared_aa"}, {"label": "2pq (Aa 基因型频率)", "unit": "", "key": "two_pq_Aa"}], "formula_str": "p+q=1; 基因型频率: p², 2pq, q²"},
    {"name": "百分误差", "subject": "生物", "func": calc_percent_error, "inputs": [{"label": "实验值", "unit": "", "key": "experimental_value"}, {"label": "理论值", "unit": "", "key": "theoretical_value"}], "outputs": [{"label": "百分误差", "unit": "%", "key": "percent_error"}], "formula_str": "百分误差 = (|实验值 - 理论值| / |理论值|) * 100%"},
    {"name": "稀释因子", "subject": "生物", "func": calc_dilution_factor, "inputs": [{"label": "初始体积", "unit": "mL (或其他)", "key": "initial_volume"}, {"label": "最终体积", "unit": "mL (同上)", "key": "final_volume"}], "outputs": [{"label": "稀释因子", "unit": "", "key": "dilution_factor"}], "formula_str": "稀释因子 = 最终体积 / 初始体积"},
    {"name": "酶比活", "subject": "生物", "func": calc_specific_activity_enzyme, "inputs": [{"label": "酶总活性", "unit": "单位", "key": "enzyme_activity_units"}, {"label": "总蛋白质量", "unit": "mg", "key": "total_protein_mass_mg"}], "outputs": [{"label": "酶比活", "unit": "单位/mg", "key": "specific_activity_units_per_mg"}], "formula_str": "比活 = 酶总活性 / 总蛋白质量"},

]


class ScientificCalculatorApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("科学公式计算器 Pro")
        self.root.geometry("850x900")

        self.input_widgets = []
        self.output_widgets = []
        self.current_formula_data_obj = None
        self.subject_formulas_map = {}
        self.calculation_history = []

        top_controls_frame = ttk.Frame(self.root)
        top_controls_frame.pack(pady=(10,5), padx=10, fill="x")

        self.subject_label_widget = ttk.Label(top_controls_frame, text="科目:")
        self.subject_label_widget.pack(side=tk.LEFT, padx=(0,2), pady=5)
        self.subject_var_for_tk = tk.StringVar()
        subjects_list = sorted(list(set(f["subject"] for f in ALL_FORMULAS)))
        self.subject_combo = ttk.Combobox(top_controls_frame, textvariable=self.subject_var_for_tk,
                                          values=subjects_list, state="readonly", width=10)
        self.subject_combo.pack(side=tk.LEFT, padx=2, pady=5)
        self.subject_combo.bind("<<ComboboxSelected>>", self.update_formula_list)

        self.formula_label_widget = ttk.Label(top_controls_frame, text="公式:")
        self.formula_label_widget.pack(side=tk.LEFT, padx=(5,2), pady=5)
        self.formula_var_for_tk = tk.StringVar()
        self.formula_combo = ttk.Combobox(top_controls_frame, textvariable=self.formula_var_for_tk,
                                          state="readonly", width=40)
        self.formula_combo.pack(side=tk.LEFT, padx=2, pady=5, expand=True, fill="x")
        self.formula_combo.bind("<<ComboboxSelected>>", self.setup_formula_interface)

        self.constants_button_widget = ttk.Button(top_controls_frame, text="常量参考", command=self.show_constants_window)
        self.constants_button_widget.pack(side=tk.LEFT, padx=(10,0), pady=5)

        self.formula_str_display_frame = ttk.LabelFrame(self.root, text="当前公式")
        self.formula_str_display_frame.pack(pady=2, padx=10, fill="x")
        self.formula_str_widget = ttk.Label(self.formula_str_display_frame, text="",
                                           wraplength=800, font=("Arial", 10, "italic"), justify=tk.LEFT)
        self.formula_str_widget.pack(pady=2, padx=5, fill="x")

        main_paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_pane = ttk.PanedWindow(main_paned_window, orient=tk.VERTICAL)
        main_paned_window.add(left_pane, weight=3)

        inputs_container_frame = ttk.Frame(left_pane)
        left_pane.add(inputs_container_frame, weight=2)
        self.canvas_inputs = tk.Canvas(inputs_container_frame)
        self.inputs_frame = ttk.LabelFrame(self.canvas_inputs, text="输入参数")
        self.scrollbar_inputs_y = ttk.Scrollbar(inputs_container_frame, orient="vertical", command=self.canvas_inputs.yview)
        self.canvas_inputs.configure(yscrollcommand=self.scrollbar_inputs_y.set)
        self.scrollbar_inputs_y.pack(side=tk.RIGHT, fill="y")
        self.canvas_inputs.pack(side=tk.LEFT, fill="both", expand=True)
        self.canvas_inputs_window = self.canvas_inputs.create_window((0, 0), window=self.inputs_frame, anchor="nw")
        self.inputs_frame.bind("<Configure>", self.on_inputs_frame_configure)
        self.canvas_inputs.bind("<Configure>", self.on_canvas_configure)

        self.outputs_outer_frame = ttk.Frame(left_pane)
        left_pane.add(self.outputs_outer_frame, weight=1)
        self.outputs_frame = ttk.LabelFrame(self.outputs_outer_frame, text="计算结果")
        self.outputs_frame.pack(fill=tk.BOTH, expand=True)

        right_pane = ttk.Frame(main_paned_window)
        main_paned_window.add(right_pane, weight=1)

        self.history_display_frame = ttk.LabelFrame(right_pane, text="计算历史")
        self.history_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=0)
        self.history_text_area = scrolledtext.ScrolledText(self.history_display_frame, wrap=tk.WORD, height=10, width=30, state=tk.DISABLED)
        self.history_text_area.pack(fill=tk.BOTH, expand=True, pady=5, padx=5)
        self.clear_history_button_widget = ttk.Button(self.history_display_frame, text="清空历史", command=self.clear_history)
        self.clear_history_button_widget.pack(pady=5)

        self.calc_button = ttk.Button(self.root, text="计算", command=self.perform_calculation, state=tk.DISABLED)
        self.calc_button.pack(pady=10)

        if subjects_list:
            self.subject_combo.current(0)
            self.update_formula_list()

    def show_constants_window(self):
        constants_win = Toplevel(self.root)
        constants_win.title("常用科学常数") # Direct Chinese
        constants_win.geometry("500x550") # Adjusted width for longer names
        constants_win.transient(self.root); constants_win.grab_set()
        tree_frame = ttk.Frame(constants_win); tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        cols = ("name", "value", "unit"); tree = ttk.Treeview(tree_frame, columns=cols, show='headings')
        tree.heading("name", text="常数名称"); tree.heading("value", text="数值"); tree.heading("unit", text="单位") # Direct Chinese
        tree.column("name", width=220, anchor=tk.W); tree.column("value", width=130, anchor=tk.E); tree.column("unit", width=120, anchor=tk.W) # Adjusted widths
        for const_data in CONSTANTS_REFERENCE_DATA:
            value_str = f"{const_data['value']:.4e}" if isinstance(const_data['value'], float) and (abs(const_data['value']) > 1e5 or (abs(const_data['value']) < 1e-3 and const_data['value']!=0) ) else str(const_data['value'])
            tree.insert("", tk.END, values=(const_data["name"], value_str, const_data["unit"])) # Uses name from CONSTANTS_REFERENCE_DATA
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_y = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar_y.set); scrollbar_y.pack(side=tk.RIGHT, fill="y")

    def add_to_history(self, formula_name_disp, inputs_disp, results_disp):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"--- {timestamp} ---\n公式: {formula_name_disp}\n输入:\n"
        for key, val_unit in inputs_disp.items(): entry += f"  {key} = {val_unit[0]} {val_unit[1]}\n"
        entry += "结果:\n"
        for key, val_unit_tuple in results_disp.items(): # Ensure tuple unpacking if val_unit is tuple
            entry += f"  {key}: {val_unit_tuple[0]} {val_unit_tuple[1]}\n"

        entry += "-" * 30 + "\n\n"
        self.calculation_history.insert(0, entry)
        if len(self.calculation_history) > 20: self.calculation_history.pop()
        self.update_history_display()

    def update_history_display(self):
        self.history_text_area.config(state=tk.NORMAL)
        self.history_text_area.delete(1.0, tk.END)
        for item in self.calculation_history: self.history_text_area.insert(tk.END, item)
        self.history_text_area.config(state=tk.DISABLED)
        self.history_text_area.yview_moveto(0)

    def clear_history(self):
        self.calculation_history.clear(); self.update_history_display()

    def update_formula_list(self, event=None):
        selected_subject = self.subject_var_for_tk.get()
        # Using 'name' directly as it's a simple string in your current ALL_FORMULAS
        subject_formulas_data = sorted([f for f in ALL_FORMULAS if f["subject"] == selected_subject], key=lambda x: x["name"])
        self.subject_formulas_map = {f["name"]: f for f in subject_formulas_data}
        formula_names = [f["name"] for f in subject_formulas_data]
        self.formula_combo["values"] = formula_names
        if formula_names:
            self.formula_var_for_tk.set(formula_names[0])
            self.setup_formula_interface()
            self.calc_button.config(state=tk.NORMAL)
        else:
            self.formula_var_for_tk.set(""); self.clear_formula_interface(); self.calc_button.config(state=tk.DISABLED)

    def on_inputs_frame_configure(self, event=None):
        if self.canvas_inputs.winfo_exists(): self.canvas_inputs.configure(scrollregion=self.canvas_inputs.bbox("all"))
    def on_canvas_configure(self, event=None):
        if self.canvas_inputs.winfo_exists() and hasattr(self,'canvas_inputs_window') and self.canvas_inputs_window : self.canvas_inputs.itemconfig(self.canvas_inputs_window, width=event.width)

    def clear_formula_interface(self):
        if hasattr(self, 'inputs_frame') and self.inputs_frame.winfo_exists():
            for child in self.inputs_frame.winfo_children(): child.destroy()
        self.input_widgets.clear()
        if hasattr(self, 'outputs_frame') and self.outputs_frame.winfo_exists():
            for child in self.outputs_frame.winfo_children(): child.destroy()
        self.output_widgets.clear()
        if hasattr(self, 'formula_str_widget') and self.formula_str_widget.winfo_exists(): self.formula_str_widget.config(text="")
        self.current_formula_data_obj = None
        if hasattr(self, 'inputs_frame') and self.inputs_frame.winfo_exists(): self.inputs_frame.columnconfigure(1, weight=0)

    def setup_formula_interface(self, event=None):
        self.clear_formula_interface()
        selected_formula_name = self.formula_var_for_tk.get()
        if not selected_formula_name: return

        self.current_formula_data_obj = self.subject_formulas_map.get(selected_formula_name)
        if not self.current_formula_data_obj: return

        self.formula_str_widget.config(text=f"{self.current_formula_data_obj.get('formula_str', 'N/A')}")
        for i_idx, item_meta in enumerate(self.current_formula_data_obj.get("inputs",[])):
            lbl_text = f"{item_meta['label']} ({item_meta['unit']}):"
            f_name_zh_stable = self.current_formula_data_obj.get("name", "N/A_ZH_NAME")
            param_key = item_meta['key']

            # Add optional hints based on Chinese names as before
            if param_key == 'temp_c' and "能斯特" in f_name_zh_stable : lbl_text += " (可选,默认25°C)"
            elif param_key == 'angle_deg' and f_name_zh_stable == "功 (W = Fdcosθ)": lbl_text += " (可选,默认0°)"
            elif param_key == 'Tf_pure_solvent_C' and "溶液凝固点" in f_name_zh_stable: lbl_text = f"{item_meta['label']} (可选,默认水{PURE_WATER_FREEZING_POINT_C}°C):" # Make sure default shown clearly
            elif param_key == 'Tb_pure_solvent_C' and "溶液沸点" in f_name_zh_stable: lbl_text = f"{item_meta['label']} (可选,默认水{PURE_WATER_BOILING_POINT_C}°C):"
            elif param_key == 'angle_deg_v_B' and "洛伦兹力" in f_name_zh_stable: lbl_text += " (可选,默认90°)"
            elif param_key == 'angle_deg_diffraction' and ("衍射" in f_name_zh_stable or "干涉" in f_name_zh_stable) : lbl_text += " (可选,默认0°)"
            elif param_key == 'molar_mass_gas_g_mol' and "光合/呼吸速率" in f_name_zh_stable : lbl_text = f"{item_meta['label']} (可选,默认O₂ {MOLAR_MASS_O2_g_mol}g/mol):"
            elif param_key == 'R_gas_const_MPa_L_mol_K' and "渗透势" in f_name_zh_stable : lbl_text = f"{item_meta['label']} (默认{GAS_CONSTANT_MPA_L}):"
            elif param_key == 'R_gas_const_L_atm_mol_K' and "渗透压" in f_name_zh_stable : lbl_text = f"{item_meta['label']} (默认{GAS_CONSTANT_L_ATM}):"
            elif param_key == 'R_gas_const_L_atm_mol_K' and "Kp 与 Kc 转换" in f_name_zh_stable : lbl_text = f"{item_meta['label']} ({GAS_CONSTANT_L_ATM} L·atm/mol·K):" # Displaying value
            elif param_key == 'i_vant_hoff' and ("渗透压" in f_name_zh_stable or "渗透势" in f_name_zh_stable) : lbl_text += " (可选,默认1)"
            elif param_key in ['stoich_coeffs', 'delta_Hf_products_kJ_mol', 'delta_Hf_reactants_kJ_mol'] and "反应热" in f_name_zh_stable:
                 lbl_text = f"{item_meta['label']}:" # Remove unit for dictionary input
            elif param_key in ['a_vdw', 'b_vdw'] and "范德华方程" in f_name_zh_stable:
                 lbl_text = f"{item_meta['label']} ({item_meta['unit']}):" # Keep unit for a and b

            lbl = ttk.Label(self.inputs_frame, text=lbl_text)
            lbl.grid(row=i_idx, column=0, padx=5, pady=3, sticky="w")

            # Special handling for dictionary inputs
            if param_key in ['stoich_coeffs', 'delta_Hf_products_kJ_mol', 'delta_Hf_reactants_kJ_mol']:
                 entry = scrolledtext.ScrolledText(self.inputs_frame, wrap=tk.WORD, height=3, width=25)
                 entry.grid(row=i_idx, column=1, padx=5, pady=3, sticky="ew")
                 # Add a hint for dictionary format
                 hint_text = ""
                 if param_key == 'stoich_coeffs': hint_text = "格式: {'产物A': 1, '反应物B': -1}"
                 elif param_key == 'delta_Hf_products_kJ_mol': hint_text = "格式: {'产物A': -393.5}"
                 elif param_key == 'delta_Hf_reactants_kJ_mol': hint_text = "格式: {'反应物B': -241.8}"
                 entry.insert(tk.END, hint_text)
            else:
                entry = ttk.Entry(self.inputs_frame, width=25)
                entry.grid(row=i_idx, column=1, padx=5, pady=3, sticky="ew")

            self.input_widgets.append((lbl, entry, param_key, item_meta['label']))

        if self.inputs_frame.winfo_exists(): self.inputs_frame.columnconfigure(1, weight=1)
        for o_idx, item_meta in enumerate(self.current_formula_data_obj.get("outputs",[])):
            lbl_desc = ttk.Label(self.outputs_frame, text=f"{item_meta['label']} ({item_meta['unit']}):")
            lbl_desc.grid(row=o_idx, column=0, padx=5, pady=3, sticky="w")
            lbl_val = ttk.Label(self.outputs_frame, text="---", width=35, relief="sunken", anchor="w", wraplength=250)
            lbl_val.grid(row=o_idx, column=1, padx=5, pady=3, sticky="ew")
            self.output_widgets.append((lbl_desc, lbl_val, item_meta['key']))
        if self.outputs_frame.winfo_exists(): self.outputs_frame.columnconfigure(1, weight=1)
        self.calc_button.config(state=tk.NORMAL)
        self.inputs_frame.update_idletasks()
        if self.canvas_inputs.winfo_exists(): self.canvas_inputs.config(scrollregion=self.canvas_inputs.bbox("all")); self.canvas_inputs.yview_moveto(0)

    def perform_calculation(self):
        if not self.current_formula_data_obj: messagebox.showerror("错误", "未选择公式!"); return

        inputs_values_dict = {}; inputs_display_for_history = {}
        try:
            for _, entry_widget, input_key, param_name_for_err_msg in self.input_widgets:
                val_str = ""
                if isinstance(entry_widget, scrolledtext.ScrolledText):
                    val_str = entry_widget.get("1.0", tk.END).strip()
                    # Remove hint text if it's still there
                    hint_texts = [
                        "格式: {'产物A': 1, '反应物B': -1}",
                        "格式: {'产物A': -393.5}",
                        "格式: {'反应物B': -241.8}"
                    ]
                    if val_str in hint_texts: val_str = ""

                else:
                    val_str = entry_widget.get().strip()

                is_optional_empty = False
                f_name_zh_stable = self.current_formula_data_obj.get("name", "N/A_ZH_NAME")
                default_val = None

                # Optional parameter handling with defaults
                if input_key == 'temp_c' and "能斯特" in f_name_zh_stable: default_val = 25.0
                elif input_key == 'angle_deg' and f_name_zh_stable == "功 (W = Fdcosθ)": default_val = 0.0
                elif input_key == 'Tf_pure_solvent_C' and "溶液凝固点" in f_name_zh_stable: default_val = PURE_WATER_FREEZING_POINT_C
                elif input_key == 'Tb_pure_solvent_C' and "溶液沸点" in f_name_zh_stable: default_val = PURE_WATER_BOILING_POINT_C
                elif input_key == 'angle_deg_v_B' and "洛伦兹力" in f_name_zh_stable: default_val = 90.0
                elif input_key == 'angle_deg_diffraction' and ("衍射" in f_name_zh_stable or "干涉" in f_name_zh_stable) : default_val = 0.0
                elif input_key == 'molar_mass_gas_g_mol' and "光合/呼吸速率" in f_name_zh_stable : default_val = MOLAR_MASS_O2_g_mol
                elif input_key == 'R_gas_const_MPa_L_mol_K' and "渗透势" in f_name_zh_stable : default_val = GAS_CONSTANT_MPA_L
                elif input_key == 'R_gas_const_L_atm_mol_K' and ("渗透压" in f_name_zh_stable or "Kp 与 Kc 转换" in f_name_zh_stable) : default_val = GAS_CONSTANT_L_ATM
                elif input_key == 'i_vant_hoff' and ("渗透压" in f_name_zh_stable or "渗透势" in f_name_zh_stable) : default_val = 1.0
                elif input_key == 'time_unit' and ("出生率" in f_name_zh_stable or "死亡率" in f_name_zh_stable) : default_val = 1.0


                if default_val is not None and not val_str:
                    inputs_values_dict[input_key] = default_val; is_optional_empty = True

                if is_optional_empty:
                    unit_text_hist = "";
                    for im in self.current_formula_data_obj.get("inputs",[]):
                         if im['key'] == input_key: unit_text_hist = im["unit"]; break
                    inputs_display_for_history[param_name_for_err_msg] = (f"{inputs_values_dict[input_key]} (默认)", unit_text_hist); continue

                if not val_str and default_val is None: messagebox.showerror("输入错误", f"参数 '{param_name_for_err_msg}' 不能为空。"); entry_widget.focus_set(); return

                # Special handling for dictionary inputs
                if input_key in ['stoich_coeffs', 'delta_Hf_products_kJ_mol', 'delta_Hf_reactants_kJ_mol']:
                    try:
                        # Safely evaluate the dictionary string
                        inputs_values_dict[input_key] = eval(val_str)
                        if not isinstance(inputs_values_dict[input_key], dict): raise ValueError("Input is not a valid dictionary.")
                        inputs_display_for_history[param_name_for_err_msg] = (val_str, "") # No unit for dict
                    except Exception as e:
                        messagebox.showerror("输入错误", f"参数 '{param_name_for_err_msg}' 的值 '{val_str}' 不是有效的字典格式。\n详细: {e}"); entry_widget.focus_set(); return
                else:
                    try:
                        float_val = float(val_str)
                        expected_int_keys = ["n_electrons_transferred", "m_order", "ni_initial_level", "nf_final_level", "Np_primary_turns", "Ns_secondary_turns"]
                        if input_key in expected_int_keys:
                            if not float_val.is_integer(): messagebox.showerror("输入错误", f"参数 '{param_name_for_err_msg}' 必须为整数。"); entry_widget.focus_set(); return
                            inputs_values_dict[input_key] = int(float_val)
                        elif input_key == "gender_male_bool":
                            if val_str.lower() in ['true', '1', 'yes', 'male', '男']: inputs_values_dict[input_key] = True
                            elif val_str.lower() in ['false', '0', 'no', 'female', '女']: inputs_values_dict[input_key] = False
                            else: messagebox.showerror("输入错误", f"参数 '{param_name_for_err_msg}' 必须是 True/False 或 1/0。"); entry_widget.focus_set(); return
                        else: inputs_values_dict[input_key] = float_val
                        unit_hist = "";
                        for im_hist in self.current_formula_data_obj.get("inputs",[]):
                            if im_hist['key'] == input_key: unit_hist = im_hist["unit"]; break
                        inputs_display_for_history[param_name_for_err_msg] = (val_str, unit_hist)
                    except ValueError: messagebox.showerror("输入错误", f"参数 '{param_name_for_err_msg}' 的值 '{val_str}' 不是有效的数字。"); entry_widget.focus_set(); return
        except Exception as e: messagebox.showerror("输入处理错误", f"获取输入时发生错误: {e}"); return

        calculation_func = self.current_formula_data_obj["func"]
        results_display_for_history = {}
        try:
            results_dict = calculation_func(**inputs_values_dict)
            if results_dict is None: messagebox.showerror("计算错误", "计算函数未返回结果。"); return
            if "error" in results_dict:
                 messagebox.showerror("计算错误", results_dict["error"]) # Using raw error message from calc function
                 for _, lbl_val_w, _key_out in self.output_widgets: lbl_val_w.config(text="---")
                 return # IMPORTANT: This return was causing the IndentationError if not properly placed after loop

            # If no error, proceed to display results
            for lbl_desc_w, lbl_val_w, output_key in self.output_widgets:
                val_to_disp_str = "---"; raw_val_for_history = "---"
                f_name_zh_res = self.current_formula_data_obj.get("name", "") # For BMI category

                # Handle dynamic output text (Snell's, Bohr, etc.)
                is_snell_calc_func = (calculation_func == calc_snells_law_theta2)
                is_bohr_calc_func = (calculation_func == calc_bohr_model_energy_transition)
                is_percent_error_func = (calculation_func == calc_percent_error)
                is_rq_func = (calculation_func == calc_respiratory_quotient)
                is_capacitance_func = (calculation_func == calc_capacitance)
                is_power_v2r_func = (calculation_func == calc_power_v2r)
                is_de_broglie_func = (calculation_func == calc_de_broglie_wavelength)
                is_half_life_second_order_func = (calculation_func == calc_half_life_second_order)
                is_henrys_law_func = (calculation_func == calc_henrys_law_solubility) # Added

                if is_snell_calc_func:
                    if output_key == "theta2_info_content" and "theta2_info_content" in results_dict:
                        val_to_disp_str = results_dict["theta2_info_content"]
                    elif output_key == "theta2_deg_val" and "theta2_deg_val" in results_dict:
                        raw_val = results_dict["theta2_deg_val"]
                        val_to_disp_str = f"{raw_val:.5g}" if isinstance(raw_val, float) else str(raw_val)
                    elif output_key == "sin_theta2_calc_val" and "sin_theta2_calc_val" in results_dict:
                        raw_val = results_dict["sin_theta2_calc_val"]
                        val_to_disp_str = f"{raw_val:.5g}" if isinstance(raw_val, float) else str(raw_val)
                elif is_bohr_calc_func:
                    if output_key == "transition_type" and "transition_type" in results_dict:
                         val_to_disp_str = results_dict["transition_type"]
                    elif output_key in results_dict: # Handle other outputs for Bohr
                         raw_val = results_dict[output_key]
                         if isinstance(raw_val, float):
                            if math.isinf(raw_val): val_to_disp_str = str(raw_val)
                            elif math.isnan(raw_val): val_to_disp_str = "NaN (无效操作)"
                            elif abs(raw_val) > 1e-3 and abs(raw_val) < 1e7: val_to_disp_str = f"{raw_val:.8g}".rstrip('0').rstrip('.') if '.' in f"{raw_val:.8g}" else f"{raw_val:.8g}"
                            else: val_to_disp_str = f"{raw_val:.4e}"
                         else: val_to_disp_str = str(raw_val)
                elif is_percent_error_func and output_key == "percent_error":
                    raw_val = results_dict.get(output_key, "---")
                    if math.isinf(raw_val): val_to_disp_str = "无限大"
                    elif math.isnan(raw_val): val_to_disp_str = "未定义"
                    elif isinstance(raw_val, (int, float)): val_to_disp_str = f"{raw_val:.4g}"
                    else: val_to_disp_str = str(raw_val)
                elif is_rq_func and output_key == "RQ":
                     raw_val = results_dict.get(output_key, "---")
                     if math.isinf(raw_val): val_to_disp_str = "无限大"
                     elif math.isnan(raw_val): val_to_disp_str = "未定义"
                     elif isinstance(raw_val, (int, float)): val_to_disp_str = f"{raw_val:.4g}"
                     else: val_to_disp_str = str(raw_val)
                elif is_capacitance_func and output_key == "C_capacitance":
                     raw_val = results_dict.get(output_key, "---")
                     if math.isinf(raw_val): val_to_disp_str = "无限大"
                     elif math.isnan(raw_val): val_to_disp_str = "未定义"
                     elif isinstance(raw_val, (int, float)): val_to_disp_str = f"{raw_val:.4g}"
                     else: val_to_disp_str = str(raw_val)
                elif is_power_v2r_func and output_key == "P_power_watt":
                     raw_val = results_dict.get(output_key, "---")
                     if math.isinf(raw_val): val_to_disp_str = "无限大"
                     elif math.isnan(raw_val): val_to_disp_str = "未定义"
                     elif isinstance(raw_val, (int, float)): val_to_disp_str = f"{raw_val:.4g}"
                     else: val_to_disp_str = str(raw_val)
                elif is_de_broglie_func and output_key == "lambda_debroglie":
                     raw_val = results_dict.get(output_key, "---")
                     if math.isinf(raw_val): val_to_disp_str = "无限大"
                     elif math.isnan(raw_val): val_to_disp_str = "未定义"
                     elif isinstance(raw_val, (int, float)): val_to_disp_str = f"{raw_val:.4g}"
                     else: val_to_disp_str = str(raw_val)
                elif is_half_life_second_order_func and output_key == "t_half_second_order_s":
                     raw_val = results_dict.get(output_key, "---")
                     if math.isinf(raw_val): val_to_disp_str = "无限大"
                     elif math.isnan(raw_val): val_to_disp_str = "未定义"
                     elif isinstance(raw_val, (int, float)): val_to_disp_str = f"{raw_val:.4g}"
                     else: val_to_disp_str = str(raw_val)
                elif is_henrys_law_func and output_key == "S_solubility_M": # Added Henry's Law check
                     raw_val = results_dict.get(output_key, "---")
                     if math.isinf(raw_val): val_to_disp_str = "无限大"
                     elif math.isnan(raw_val): val_to_disp_str = "未定义"
                     elif isinstance(raw_val, (int, float)): val_to_disp_str = f"{raw_val:.4g}"
                     else: val_to_disp_str = str(raw_val)

                elif output_key in results_dict: # General case for standard outputs
                    raw_val = results_dict[output_key]
                    raw_val_for_history = raw_val
                    if isinstance(raw_val, float):
                        if math.isinf(raw_val): val_to_disp_str = str(raw_val)
                        elif math.isnan(raw_val): val_to_disp_str = "NaN (无效操作)"
                        elif (0.00001 <= abs(raw_val) < 1e7 or raw_val == 0) and not ('e' in f"{raw_val:.8g}" or 'E' in f"{raw_val:.8g}"):
                            s = f"{raw_val:.8g}".rstrip('0').rstrip('.') if '.' in f"{raw_val:.8g}" else f"{raw_val:.8g}"
                            val_to_disp_str = s
                        else: val_to_disp_str = f"{raw_val:.4e}"
                    elif output_key == "Category" and "BMI" in f_name_zh_res : val_to_disp_str = raw_val
                    else: val_to_disp_str = str(raw_val)

                lbl_val_w.config(text=val_to_disp_str)
                hist_val_entry = val_to_disp_str
                if isinstance(raw_val_for_history, float) and not (math.isinf(raw_val_for_history) or math.isnan(raw_val_for_history)):
                     hist_val_entry = f"{raw_val_for_history:.6g}"
                elif raw_val_for_history == "---" and val_to_disp_str != "---":
                     hist_val_entry = val_to_disp_str

                # Get the unit for history display
                unit_hist_out = ""
                for om_hist in self.current_formula_data_obj.get("outputs",[]):
                    if om_hist['key'] == output_key: unit_hist_out = om_hist["unit"]; break

                results_display_for_history[lbl_desc_w.cget("text").replace(":","")] = (hist_val_entry, unit_hist_out)


                if "info" in results_dict and isinstance(results_dict["info"], str) and output_key == self.output_widgets[0][2]:
                     info_text = results_dict["info"]
                     current_text = lbl_val_w.cget("text")
                     full_text = f"{current_text} [{info_text}]" if current_text != "---" else f"[{info_text}]"
                     lbl_val_w.config(text=full_text)
                     # Update history with info text for the first output
                     if results_display_for_history :
                         first_hist_key = list(results_display_for_history.keys())[0]
                         results_display_for_history[first_hist_key] = (full_text, results_display_for_history[first_hist_key][1])


            self.add_to_history(self.formula_var_for_tk.get(), inputs_display_for_history, results_display_for_history)

        except TypeError as te:
            messagebox.showerror("内部错误", f"公式 '{self.current_formula_data_obj['name']}' 的参数配置或调用错误。\n详细: {te}")
            print(f"TypeError: {te} for {calculation_func.__name__}. Expected: {inspect.signature(calculation_func)}. Got: {inputs_values_dict}")
        except Exception as e:
            messagebox.showerror("计算错误", f"计算中未知错误: {type(e).__name__} - {e}")
            for _, lbl_val_w, _ in self.output_widgets: lbl_val_w.config(text="---")


if __name__ == "__main__":
    main_window = tk.Tk()
    app = ScientificCalculatorApp(main_window)
    main_window.mainloop()
