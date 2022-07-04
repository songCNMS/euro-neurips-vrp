import numpy as np
import elkai
from cvrptw import read_input_cvrptw

def get_tsp_solution(nb_customers, distance_warehouses, distance_matrix):
    M = np.zeros((nb_customers+1, nb_customers+1))
    for i in range(nb_customers+1):
        for j in range(nb_customers+1):
            if i == 0 and j == 0: continue
            elif i == 0: M[i, j] = distance_warehouses[j-1]
            elif j == 0: M[i, j] = distance_warehouses[i-1]
            else: M[i, j] = distance_matrix[i-1][j-1]
    route = elkai.solve_float_matrix(M)[1:]
    return [c-1 for c in route]

def cus_route_cost(distance_warehouses, distance_matrix):
    route = """path_name: path_name: PATH_114, ['Customer_115']
path_name: PATH_400, ['Customer_354', 'Customer_150', 'Customer_83', 'Customer_217', 'Customer_209', 'Customer_185', 'Customer_10', 'Customer_152', 'Customer_336', 'Customer_286']
path_name: PATH_401, ['Customer_88', 'Customer_392', 'Customer_36', 'Customer_191', 'Customer_349', 'Customer_344', 'Customer_91', 'Customer_130', 'Customer_372', 'Customer_184', 'Customer_99']
path_name: PATH_402, ['Customer_366', 'Customer_301', 'Customer_139', 'Customer_280', 'Customer_143', 'Customer_215', 'Customer_46', 'Customer_147', 'Customer_153', 'Customer_89', 'Customer_166', 'Customer_258']
path_name: PATH_403, ['Customer_176', 'Customer_114', 'Customer_348', 'Customer_235', 'Customer_388', 'Customer_281', 'Customer_129', 'Customer_263', 'Customer_175', 'Customer_3']
path_name: PATH_404, ['Customer_44', 'Customer_5', 'Customer_160', 'Customer_337', 'Customer_330', 'Customer_93', 'Customer_229', 'Customer_295', 'Customer_103', 'Customer_122', 'Customer_137', 'Customer_384']
path_name: PATH_405, ['Customer_303', 'Customer_144', 'Customer_24', 'Customer_199', 'Customer_82', 'Customer_296', 'Customer_359', 'Customer_254', 'Customer_293', 'Customer_60', 'Customer_53', 'Customer_313']
path_name: PATH_406, ['Customer_335', 'Customer_204', 'Customer_385', 'Customer_221', 'Customer_284', 'Customer_351', 'Customer_58', 'Customer_356', 'Customer_107', 'Customer_315', 'Customer_80', 'Customer_86']
path_name: PATH_407, ['Customer_23', 'Customer_322', 'Customer_151', 'Customer_121', 'Customer_213', 'Customer_294', 'Customer_120', 'Customer_50', 'Customer_383', 'Customer_261', 'Customer_131']
path_name: PATH_408, ['Customer_205', 'Customer_324', 'Customer_338', 'Customer_161', 'Customer_233', 'Customer_193', 'Customer_19', 'Customer_55', 'Customer_270', 'Customer_134']
path_name: PATH_409, ['Customer_63', 'Customer_156', 'Customer_133', 'Customer_39', 'Customer_333', 'Customer_382', 'Customer_256', 'Customer_62', 'Customer_65', 'Customer_32']
path_name: PATH_410, ['Customer_371', 'Customer_374', 'Customer_38', 'Customer_210', 'Customer_67', 'Customer_111', 'Customer_194', 'Customer_363', 'Customer_251', 'Customer_316', 'Customer_276', 'Customer_142']
path_name: PATH_411, ['Customer_51', 'Customer_181', 'Customer_1', 'Customer_116', 'Customer_127', 'Customer_245', 'Customer_396', 'Customer_242', 'Customer_259', 'Customer_307', 'Customer_264']
path_name: PATH_412, ['Customer_73', 'Customer_30', 'Customer_298', 'Customer_87', 'Customer_70', 'Customer_302', 'Customer_201', 'Customer_25', 'Customer_96', 'Customer_49']
path_name: PATH_413, ['Customer_16', 'Customer_136', 'Customer_317', 'Customer_265', 'Customer_380', 'Customer_85', 'Customer_328', 'Customer_177', 'Customer_95']
path_name: PATH_414, ['Customer_112', 'Customer_135', 'Customer_140', 'Customer_314', 'Customer_279', 'Customer_291', 'Customer_22', 'Customer_399', 'Customer_132']
path_name: PATH_415, ['Customer_33', 'Customer_255', 'Customer_29', 'Customer_206', 'Customer_353', 'Customer_90', 'Customer_393', 'Customer_196', 'Customer_321', 'Customer_28', 'Customer_249']
path_name: PATH_417, ['Customer_287', 'Customer_52', 'Customer_31', 'Customer_362', 'Customer_108', 'Customer_248', 'Customer_171', 'Customer_243', 'Customer_239', 'Customer_192', 'Customer_27']
path_name: PATH_419, ['Customer_182', 'Customer_236', 'Customer_61', 'Customer_11', 'Customer_369', 'Customer_110', 'Customer_35', 'Customer_223', 'Customer_218', 'Customer_234']
path_name: PATH_420, ['Customer_128', 'Customer_290', 'Customer_252', 'Customer_370', 'Customer_274', 'Customer_138', 'Customer_170', 'Customer_4', 'Customer_190', 'Customer_275', 'Customer_240']
path_name: PATH_421, ['Customer_56', 'Customer_141', 'Customer_364', 'Customer_340', 'Customer_373', 'Customer_167', 'Customer_148', 'Customer_378', 'Customer_164', 'Customer_367', 'Customer_391']
path_name: PATH_424, ['Customer_94', 'Customer_54', 'Customer_180', 'Customer_179', 'Customer_386', 'Customer_379', 'Customer_312', 'Customer_246', 'Customer_81', 'Customer_197', 'Customer_84']
path_name: PATH_425, ['Customer_92', 'Customer_310', 'Customer_266', 'Customer_165', 'Customer_308', 'Customer_146', 'Customer_268', 'Customer_79', 'Customer_339', 'Customer_41']
path_name: PATH_426, ['Customer_332', 'Customer_387', 'Customer_262', 'Customer_400', 'Customer_8', 'Customer_319', 'Customer_260', 'Customer_43', 'Customer_34', 'Customer_230', 'Customer_224']
path_name: PATH_427, ['Customer_278', 'Customer_78', 'Customer_231', 'Customer_188', 'Customer_125', 'Customer_225', 'Customer_323', 'Customer_18', 'Customer_334']
path_name: PATH_428, ['Customer_358', 'Customer_216', 'Customer_203', 'Customer_365', 'Customer_195', 'Customer_285', 'Customer_341', 'Customer_207', 'Customer_187', 'Customer_200']
path_name: PATH_429, ['Customer_397', 'Customer_72', 'Customer_42', 'Customer_389', 'Customer_173', 'Customer_220', 'Customer_183', 'Customer_123', 'Customer_232', 'Customer_75', 'Customer_228', 'Customer_273']
path_name: PATH_430, ['Customer_109', 'Customer_292', 'Customer_326', 'Customer_226', 'Customer_172', 'Customer_117', 'Customer_26', 'Customer_350', 'Customer_282', 'Customer_352', 'Customer_40', 'Customer_119']
path_name: PATH_431, ['Customer_17', 'Customer_214', 'Customer_355', 'Customer_247', 'Customer_162', 'Customer_238', 'Customer_320', 'Customer_168', 'Customer_283', 'Customer_289', 'Customer_158', 'Customer_104', 'Customer_227']
path_name: PATH_432, ['Customer_186', 'Customer_154', 'Customer_381', 'Customer_395', 'Customer_361', 'Customer_272', 'Customer_306', 'Customer_377', 'Customer_124', 'Customer_2', 'Customer_347', 'Customer_376']
path_name: PATH_433, ['Customer_155', 'Customer_118', 'Customer_309', 'Customer_145', 'Customer_318', 'Customer_329', 'Customer_297', 'Customer_21', 'Customer_346', 'Customer_45', 'Customer_106']
path_name: PATH_434, ['Customer_343', 'Customer_269', 'Customer_64', 'Customer_208', 'Customer_398', 'Customer_66', 'Customer_331', 'Customer_305', 'Customer_163', 'Customer_100']
path_name: PATH_435, ['Customer_126', 'Customer_97', 'Customer_375', 'Customer_149', 'Customer_101', 'Customer_7', 'Customer_325', 'Customer_368', 'Customer_37', 'Customer_20']
path_name: PATH_436, ['Customer_105', 'Customer_202', 'Customer_342', 'Customer_277', 'Customer_257', 'Customer_300', 'Customer_13', 'Customer_15', 'Customer_76', 'Customer_311', 'Customer_241']
path_name: PATH_453, ['Customer_360', 'Customer_9', 'Customer_12', 'Customer_157', 'Customer_47', 'Customer_299', 'Customer_169', 'Customer_237', 'Customer_219', 'Customer_198', 'Customer_212']
path_name: PATH_455, ['Customer_189', 'Customer_390', 'Customer_271', 'Customer_244', 'Customer_57', 'Customer_48', 'Customer_345', 'Customer_98', 'Customer_250', 'Customer_174']
path_name: PATH_459, ['Customer_102', 'Customer_69', 'Customer_222', 'Customer_159', 'Customer_304', 'Customer_71', 'Customer_113', 'Customer_211', 'Customer_59', 'Customer_253']
path_name: PATH_460, ['Customer_267', 'Customer_357', 'Customer_77', 'Customer_74', 'Customer_394', 'Customer_68', 'Customer_14', 'Customer_288', 'Customer_6', 'Customer_178', 'Customer_327']""".split("\n")
    routes = []
    for r in route:
        s = r.index('[')
        e = r.index(']')
        routes.append([int(c.split('_')[1])-1 for c in r[s+1:e].replace('\'','').split(',')])
    total_cost = 0.0
    for route in routes:
        total_cost += (distance_warehouses[route[0]] + distance_warehouses[route[-1]])
        for i in range(len(route)-1): total_cost += distance_matrix[route[i]][route[i+1]]
    return len(routes), total_cost

def compute_route_cost(distance_warehouses, distance_matrix):
    route = """Route 1 : 102 69 222 159 304 71 113 211 59 253 364 
Route 2 : 357 77 74 394 68 14 288 6 178 327 386 312 43 
Route 3 : 94 54 180 179 146 268 79 339 34 230 260 329 
Route 4 : 310 165 262 400 379 319 8 81 197 246 84 308 
Route 5 : 387 278 78 231 188 125 225 323 224 41 
Route 6 : 332 216 358 195 285 18 238 334 100 
Route 7 : 397 72 203 365 341 207 187 200 119 232 42 106 
Route 8 : 186 154 381 202 342 309 118 21 346 45 
Route 9 : 292 226 109 172 26 350 282 352 40 183 123 75 228 273 
Route 10 : 17 326 214 355 117 247 162 320 168 283 289 158 104 227 
Route 11 : 105 155 395 361 272 306 377 124 2 347 376 
Route 12 : 286 88 217 209 185 10 152 336 297 163 
Route 13 : 343 269 64 208 145 318 175 3 184 99 354 
Route 14 : 5 160 337 330 143 215 46 147 153 89 166 258 
Route 15 : 114 348 235 388 281 129 263 176 137 384 44 36 
Route 16 : 303 144 24 93 229 204 295 103 220 122 86 139 
Route 17 : 335 385 199 82 296 359 254 293 60 53 313 315 
Route 18 : 322 221 284 351 191 349 130 372 91 344 
Route 19 : 63 121 233 173 161 120 134 107 
Route 20 : 366 150 83 193 19 55 50 383 131 261 213 294 
Route 21 : 324 156 133 210 382 338 256 205 270 151 58 
Route 22 : 23 371 38 333 111 194 363 251 316 276 142 356 80 
Route 23 : 374 181 389 1 116 127 245 65 62 32 264 
Route 24 : 298 87 39 67 307 396 242 259 51 
Route 25 : 112 135 70 302 201 25 96 49 353 
Route 26 : 301 280 392 169 237 321 291 16 136 317 265 73 
Route 27 : 33 52 380 85 328 177 95 30 
Route 28 : 375 149 101 206 257 300 13 15 76 311 7 241 
Route 29 : 360 12 314 279 90 196 399 132 22 140 
Route 30 : 97 255 29 157 47 299 305 219 198 249 28 
Route 31 : 287 31 362 108 248 171 243 239 192 128 27 393 
Route 32 : 189 174 390 271 244 57 48 345 98 250 234 
Route 33 : 267 141 252 370 274 138 170 4 190 275 240 9 212 
Route 34 : 56 182 236 61 218 11 369 110 35 223 290 
Route 35 : 92 266 398 66 277 325 368 37 20 331 
Route 36 : 126 115 340 373 167 148 378 164 367 391 """.split("\n")
    routes = [r.split()[3:] for r in route]
    routes = [[int(c)-1 for c in r] for r in routes]
    total_cost = 0.0
    for route in routes:
        total_cost += (distance_warehouses[route[0]] + distance_warehouses[route[-1]])
        for i in range(len(route)-1): total_cost += distance_matrix[route[i]][route[i+1]]
    return len(routes), total_cost
            

if __name__ == '__main__':
    problem_file = "/home/lesong/cvrptw/cvrp_benchmarks/homberger_400_customer_instances/C1_4_2.TXT"
    (nb_customers, nb_trucks, truck_capacity, distance_matrix, distance_warehouses, demands, service_time,
                earliest_start, latest_end, max_horizon, warehouse_x, warehouse_y, customers_x, customers_y) = read_input_cvrptw(problem_file)

    # print(get_tsp_solution(nb_customers, distance_warehouses, distance_matrix))
    print(compute_route_cost(distance_warehouses, distance_matrix))
    print(cus_route_cost(distance_warehouses, distance_matrix))
    