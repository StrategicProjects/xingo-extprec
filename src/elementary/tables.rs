//! The literal tables of Chapter 11.11 (private). Every double-double entry
//! is the pair `(RN(c), RN(c − RN(c)))` of the value `c` named in the
//! comment of its table, as the bits `(hi, lo)`; every `f64` is the value
//! named, rounded to nearest. They were generated with `dashu-float` at
//! 2 400 bits and are checked entry by entry, bit for bit, by the unit test
//! `tables_are_the_nearest_double_doubles` of the parent module against two
//! independent oracles.

/// `ln 2 / 64` in three parts of 53 bits: `RN(c)`, `RN(c − p₁)`, `RN(c − p₁ − p₂)` (Algorithm E).
pub(super) const LN2_64: [f64; 3] = [
    f64::from_bits(0x3f86_2e42_fefa_39ef),
    f64::from_bits(0x3c1a_bc9e_3b39_803f),
    f64::from_bits(0x38a7_b57a_079a_1934),
];
/// `RN(64 / ln 2)`: only chooses the index `N` of Algorithm E (any nearby value works).
pub(super) const INV_LN2_64: f64 = f64::from_bits(0x4057_1547_652b_82fe);
/// `ln 2` in three parts: `A` rounded to 42 bits (so that `e·A` is exact for `|e| < 2¹¹`), then `RN(ln 2 − A)`, `RN(ln 2 − A − B)` (Algorithm L, step L6).
pub(super) const LN2_SPLIT: [f64; 3] = [
    f64::from_bits(0x3fe6_2e42_fefa_3800),
    f64::from_bits(0x3d2e_f357_93c7_6730),
    f64::from_bits(0x398f_97b5_7a07_9a19),
];
/// `2^(j/64)`, `j = 0, …, 63` (Algorithm E).
pub(super) const EXP2_64: [(u64, u64); 64] = [
    (0x3ff0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3ff0_2c9a_3e77_8061, 0xbc71_9083_535b_085d),
    (0x3ff0_59b0_d315_8574, 0x3c8d_73e2_a475_b465),
    (0x3ff0_8745_1875_9bc8, 0x3c61_86be_4bb2_84ff),
    (0x3ff0_b558_6cf9_890f, 0x3c98_a62e_4adc_610b),
    (0x3ff0_e3ec_32d3_d1a2, 0x3c40_3a17_27c5_7b53),
    (0x3ff1_1301_d012_5b51, 0xbc96_c510_3944_9b3a),
    (0x3ff1_429a_aea9_2de0, 0xbc93_2fbf_9af1_369e),
    (0x3ff1_72b8_3c7d_517b, 0xbc81_9041_b9d7_8a76),
    (0x3ff1_a35b_eb6f_cb75, 0x3c8e_5b4c_7b49_68e4),
    (0x3ff1_d487_3168_b9aa, 0x3c9e_016e_00a2_643c),
    (0x3ff2_063b_8862_8cd6, 0x3c8d_c775_814a_8495),
    (0x3ff2_387a_6e75_6238, 0x3c99_b07e_b6c7_0573),
    (0x3ff2_6b45_65e2_7cdd, 0x3c82_bd33_9940_e9d9),
    (0x3ff2_9e9d_f51f_dee1, 0x3c86_12e8_afad_1255),
    (0x3ff2_d285_a6e4_030b, 0x3c90_0247_54db_41d5),
    (0x3ff3_06fe_0a31_b715, 0x3c86_f46a_d231_82e4),
    (0x3ff3_3c08_b264_16ff, 0x3c93_2721_8436_59a6),
    (0x3ff3_71a7_373a_a9cb, 0xbc96_3aea_bf42_eae2),
    (0x3ff3_a7db_34e5_9ff7, 0xbc75_e436_d661_f5e3),
    (0x3ff3_dea6_4c12_3422, 0x3c8a_da09_11f0_9ebc),
    (0x3ff4_160a_21f7_2e2a, 0xbc5e_f369_1c30_9278),
    (0x3ff4_4e08_6061_892d, 0x3c48_9b7a_04ef_80d0),
    (0x3ff4_86a2_b5c1_3cd0, 0x3c73_c1a3_b690_62f0),
    (0x3ff4_bfda_d536_2a27, 0x3c7d_4397_afec_42e2),
    (0x3ff4_f9b2_769d_2ca7, 0xbc94_b309_d259_57e3),
    (0x3ff5_342b_569d_4f82, 0xbc80_7abe_1db1_3cad),
    (0x3ff5_6f47_36b5_27da, 0x3c99_bb2c_011d_93ad),
    (0x3ff5_ab07_dd48_5429, 0x3c96_324c_0546_47ad),
    (0x3ff5_e76f_15ad_2148, 0x3c9b_a6f9_3080_e65e),
    (0x3ff6_247e_b03a_5585, 0xbc93_83c1_7e40_b497),
    (0x3ff6_6238_8255_2225, 0xbc9b_b609_8759_1c34),
    (0x3ff6_a09e_667f_3bcd, 0xbc9b_dd34_13b2_6456),
    (0x3ff6_dfb2_3c65_1a2f, 0xbc6b_be3a_683c_88ab),
    (0x3ff7_1f75_e8ec_5f74, 0xbc81_6e47_8688_7a99),
    (0x3ff7_5feb_5642_67c9, 0xbc90_2459_5731_6dd3),
    (0x3ff7_a114_73eb_0187, 0xbc84_1577_ee04_992f),
    (0x3ff7_e2f3_36cf_4e62, 0x3c70_5d02_ba15_797e),
    (0x3ff8_2589_994c_ce13, 0xbc9d_4c1d_d415_32d8),
    (0x3ff8_68d9_9b44_92ed, 0xbc9f_c6f8_9bd4_f6ba),
    (0x3ff8_ace5_422a_a0db, 0x3c96_e9f1_5686_4b27),
    (0x3ff8_f1ae_9915_7736, 0x3c85_cc13_a2e3_976c),
    (0x3ff9_3737_b0cd_c5e5, 0xbc67_5fc7_81b5_7ebc),
    (0x3ff9_7d82_9fde_4e50, 0xbc9d_185b_7c1b_85d1),
    (0x3ff9_c491_82a3_f090, 0x3c7c_7c46_b071_f2be),
    (0x3ffa_0c66_7b5d_e565, 0xbc93_5949_5d1c_d533),
    (0x3ffa_5503_b23e_255d, 0xbc9d_2f6e_db8d_41e1),
    (0x3ffa_9e6b_5579_fdbf, 0x3c90_fac9_0ef7_fd31),
    (0x3ffa_e89f_995a_d3ad, 0x3c97_a1cd_345d_cc81),
    (0x3ffb_33a2_b84f_15fb, 0xbc62_805e_3084_d708),
    (0x3ffb_7f76_f2fb_5e47, 0xbc75_584f_7e54_ac3b),
    (0x3ffb_cc1e_904b_c1d2, 0x3c82_3dd0_7a2d_9e84),
    (0x3ffc_199b_dd85_529c, 0x3c81_1065_8950_48dd),
    (0x3ffc_67f1_2e57_d14b, 0x3c92_884d_ff48_3cad),
    (0x3ffc_b720_dcef_9069, 0x3c75_03cb_d1e9_49db),
    (0x3ffd_072d_4a07_897c, 0xbc9c_bc37_4379_7a9c),
    (0x3ffd_5818_dcfb_a487, 0x3c82_ed02_d75b_3707),
    (0x3ffd_a9e6_03db_3285, 0x3c9c_2300_696d_b532),
    (0x3ffd_fc97_337b_9b5f, 0xbc91_a5cd_4f18_4b5c),
    (0x3ffe_502e_e78b_3ff6, 0x3c83_9e89_80a9_cc8f),
    (0x3ffe_a4af_a2a4_90da, 0xbc9e_9c23_179c_2893),
    (0x3ffe_fa1b_ee61_5a27, 0x3c9d_c7f4_86a4_b6b0),
    (0x3fff_5076_5b6e_4540, 0x3c99_d3e1_2dd8_a18b),
    (0x3fff_a7c1_819e_90d8, 0x3c87_4853_f3a5_931e),
];
/// `2^(j/64) − 1`, `j = 0, …, 63` (Algorithm M, large arguments).
pub(super) const EXP2_64_M1: [(u64, u64); 64] = [
    (0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3f86_4d1f_3bc0_3077, 0x3c1b_df2b_293d_e8a7),
    (0x3f96_6c34_c561_5d0f, 0xbc31_83ab_7149_735c),
    (0x3fa0_e8a3_0eb3_7901, 0x3c28_6be4_bb28_4ff4),
    (0x3fa6_ab0d_9f31_21ec, 0x3c44_c5c9_5b8c_2155),
    (0x3fac_7d86_5a7a_3440, 0x3c40_3a17_27c5_7b53),
    (0x3fb1_301d_0125_b50a, 0x3c53_aefc_6bb6_4c63),
    (0x3fb4_29aa_ea92_ddfb, 0x3c4a_080c_a1d9_2c37),
    (0x3fb7_2b83_c7d5_17ae, 0xbc49_041b_9d78_a75b),
    (0x3fba_35be_b6fc_b754, 0xbc4a_4b38_4b69_71be),
    (0x3fbd_4873_168b_9aa8, 0xbc5f_e91f_f5d9_bc3e),
    (0x3fc0_31dc_4314_66b2, 0xbc51_c453_f5ab_db59),
    (0x3fc1_c3d3_73ab_11c3, 0x3c5b_07eb_6c70_572d),
    (0x3fc3_5a2b_2f13_e6e9, 0x3c55_e99c_ca07_4ec9),
    (0x3fc4_f4ef_a8fe_f709, 0x3c68_4ba2_beb4_4954),
    (0x3fc6_942d_3720_185a, 0x3be2_3aa6_da0e_a709),
    (0x3fc8_37f0_518d_b8a9, 0x3c6b_d1ab_48c6_0b91),
    (0x3fc9_e045_9320_b7fa, 0x3c69_390c_21b2_cd2d),
    (0x3fcb_8d39_b9d5_4e55, 0x3c5c_5154_0bd1_51e6),
    (0x3fcd_3ed9_a72c_ffb7, 0x3c64_3792_533c_143a),
    (0x3fce_f532_6091_a112, 0xbc64_97db_b83d_8512),
    (0x3fd0_5828_87dc_b8a8, 0xbc5e_f369_1c30_9278),
    (0x3fd1_3821_8186_24b4, 0x3c48_9b7a_04ef_80d0),
    (0x3fd2_1a8a_d704_f340, 0x3c73_c1a3_b690_62f0),
    (0x3fd2_ff6b_54d8_a89c, 0x3c7d_4397_afec_42e2),
    (0x3fd3_e6c9_da74_b29b, 0xbc72_cc27_4965_5f8c),
    (0x3fd4_d0ad_5a75_3e07, 0x3c7f_0a83_c49d_86a6),
    (0x3fd5_bd1c_dad4_9f6a, 0xbc79_134f_fb89_b14c),
    (0x3fd6_ac1f_7521_50a5, 0x3c78_c930_1519_1eb3),
    (0x3fd7_9dbc_56b4_8522, 0xbc71_641b_3dfc_668a),
    (0x3fd8_91fa_c0e9_5613, 0xbc6c_1e0b_f205_a4b8),
    (0x3fd9_88e2_0954_8892, 0x3c71_27d9_e29b_8f31),
    (0x3fda_8279_99fc_ef32, 0x3c70_8b2f_b136_6ea9),
    (0x3fdb_7ec8_f194_68bc, 0xbc6b_be3a_683c_88ab),
    (0x3fdc_7dd7_a3b1_7dcf, 0x3c7d_2370_f2ef_0acd),
    (0x3fdd_7fad_5909_9f23, 0xbc22_2cab_98b6_e9b2),
    (0x3fde_8451_cfac_061b, 0x3c77_d510_23f6_cda2),
    (0x3fdf_8bcc_db3d_3988, 0x3c70_5d02_ba15_797e),
    (0x3fe0_4b13_3299_9c25, 0x3c65_9f11_5f56_6940),
    (0x3fe0_d1b3_3689_25d9, 0x3c2c_83b2_1584_a2e1),
    (0x3fe1_59ca_8455_41b7, 0xbc82_2c1d_52f3_69b2),
    (0x3fe1_e35d_322a_ee6c, 0x3c85_cc13_a2e3_976c),
    (0x3fe2_6e6f_619b_8bca, 0xbc67_5fc7_81b5_7ebc),
    (0x3fe2_fb05_3fbc_9c9f, 0x3c67_3d24_1f23_d17b),
    (0x3fe3_8923_0547_e120, 0x3c7c_7c46_b071_f2be),
    (0x3fe4_18cc_f6bb_cac9, 0x3c89_4d6d_45c6_559a),
    (0x3fe4_aa07_647c_4ab9, 0x3c66_8489_2395_f0f8),
    (0x3fe5_3cd6_aaf3_fb7f, 0xbc8e_0a6d_e210_059e),
    (0x3fe5_d13f_32b5_a75b, 0xbc80_bc65_9744_66fd),
    (0x3fe6_6745_709e_2bf6, 0xbc62_805e_3084_d708),
    (0x3fe6_feed_e5f6_bc8e, 0xbc75_584f_7e54_ac3b),
    (0x3fe7_983d_2097_83a4, 0x3c82_3dd0_7a2d_9e84),
    (0x3fe8_3337_bb0a_a538, 0x3c81_1065_8950_48dd),
    (0x3fe8_cfe2_5caf_a297, 0xbc8a_ef64_016f_86a6),
    (0x3fe9_6e41_b9df_20d2, 0x3c75_03cb_d1e9_49db),
    (0x3fea_0e5a_940f_12f7, 0x3c6a_1e45_e434_2b1c),
    (0x3fea_b031_b9f7_490e, 0x3c82_ed02_d75b_3707),
    (0x3feb_53cc_07b6_650b, 0xbc6e_e7fc_b492_566d),
    (0x3feb_f92e_66f7_36bd, 0x3c8c_b465_61cf_6949),
    (0x3fec_a05d_cf16_7fec, 0x3c83_9e89_80a9_cc8f),
    (0x3fed_495f_4549_21b3, 0x3c56_3dce_863d_76cc),
    (0x3fed_f437_dcc2_b44f, 0xbc61_c05b_cada_4a7c),
    (0x3fee_a0ec_b6dc_8a81, 0xbc78_b07b_489d_79d4),
    (0x3fef_4f83_033d_21b0, 0x3c87_4853_f3a5_931e),
];
/// `e^(N/64) − 1`, `N = −31, …, 31` (index `N + 31`; Algorithm M, small arguments).
pub(super) const EXPM1_64: [(u64, u64); 63] = [
    (0xbfd8_921c_445f_4add, 0x3c61_8bc8_65c7_8e3b),
    (0xbfd7_f327_a018_ddb2, 0xbc61_a901_b1ee_2bd6),
    (0xbfd7_51b2_2af6_08f0, 0xbc70_a024_9fec_9629),
    (0xbfd6_adb1_cd92_05ee, 0xbc74_edd8_a92e_b584),
    (0xbfd6_071c_47d9_53b2, 0xbc7a_771b_0577_09c4),
    (0xbfd5_5de7_3065_b4df, 0x3c65_72ec_15fe_c83d),
    (0xbfd4_b207_f3d7_9870, 0xbc67_59d4_ac13_0208),
    (0xbfd4_0373_d42c_e2e3, 0x3c75_3d8d_b804_c224),
    (0xbfd3_521f_e815_0d2b, 0xbc6b_0648_a5e3_4bb6),
    (0xbfd2_9e01_1a42_8ec6, 0xbc6d_abf5_975c_0c02),
    (0xbfd1_e70c_28b9_87f3, 0x3c74_e91d_bb17_34bd),
    (0xbfd1_2d35_a41b_a104, 0x3c63_445f_7544_e0ef),
    (0xbfd0_7071_eef1_1388, 0xbc30_9aa6_8255_3231),
    (0xbfcf_616a_79dd_a3a8, 0xbc66_b2ea_b630_20c1),
    (0xbfcd_dbe7_2473_82af, 0xbc43_1eb1_3933_e894),
    (0xbfcc_5041_854d_f7d4, 0xbc67_97d4_686c_5393),
    (0xbfca_be60_e1f2_1836, 0xbc36_f8b8_2e65_3e2d),
    (0xbfc9_262c_1c34_30a1, 0xbc64_6ff6_ec4a_4251),
    (0xbfc7_8789_b0a5_e0c0, 0x3c5e_3a6b_daec_e8f9),
    (0xbfc5_e25f_b4fd_e211, 0x3c06_4eec_8291_5df3),
    (0xbfc4_3693_d679_612d, 0xbc69_da94_a869_862a),
    (0xbfc2_840b_5836_cf67, 0xbc68_5405_051e_b425),
    (0xbfc0_caab_118a_1278, 0x3c26_ad4c_3534_65b0),
    (0xbfbe_14ae_d893_eef4, 0x3c4e_1f58_934f_97af),
    (0xbfba_85e8_c62d_9c13, 0xbc5a_df77_45e7_7188),
    (0xbfb6_e8ca_ff34_1fea, 0xbc59_573d_ed78_88b2),
    (0xbfb3_3d1b_b17d_f2e7, 0xbc4e_19c8_73b1_d6a8),
    (0xbfaf_0540_438f_d5c3, 0xbc2a_1ce0_1f9f_6ca7),
    (0xbfa7_7239_5013_0405, 0x3c2c_677a_d8fa_478d),
    (0xbf9f_8152_aee9_450e, 0x3c24_b00a_bf97_7627),
    (0xbf8f_c055_0044_16db, 0xbc28_2ef4_22ab_152a),
    (0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3f90_202a_d577_8e46, 0xbc15_1e6d_305b_eec6),
    (0x3fa0_40ac_0224_fd93, 0x3c2c_17a1_0757_5019),
    (0x3fa8_9246_d053_d178, 0x3c44_967f_31eb_2595),
    (0x3fb0_82b5_77d3_4ed8, 0xbc45_272f_f30e_ed1b),
    (0x3fb4_cd4f_c989_cd64, 0x3c55_57a8_671b_89e7),
    (0x3fb9_2937_074e_0cd7, 0xbc5d_b0b9_cc91_5fc5),
    (0x3fbd_96b0_eff0_e794, 0xbc47_5385_b2cd_f93d),
    (0x3fc1_0b02_2db7_ae68, 0xbc58_c4a5_df1e_c7e5),
    (0x3fc3_53bc_9fb0_0b21, 0x3c66_bae6_1801_1342),
    (0x3fc5_a5ac_59b9_63cb, 0xbc6f_d913_07e7_4c50),
    (0x3fc8_00f6_7b00_d7b8, 0x3c27_ab91_2c69_ffeb),
    (0x3fca_65c0_b85a_c1a9, 0x3c6a_9c18_9196_f8cd),
    (0x3fcc_d431_5e9e_0833, 0xbc21_72c3_1a17_81f1),
    (0x3fcf_4c6f_5508_ee5d, 0x3c64_6ef7_b808_180a),
    (0x3fd0_e751_0fd7_c564, 0xbc71_c5b2_e873_5a43),
    (0x3fd2_2d78_f0fa_061a, 0xbc78_9843_c496_4554),
    (0x3fd3_78c3_b084_7980, 0x3c73_b522_3eca_1712),
    (0x3fd4_c946_033e_b3de, 0xbc73_5d26_7d66_dc96),
    (0x3fd6_1f14_f169_ebc1, 0xbc78_9e2d_87fd_0d92),
    (0x3fd7_7a45_d811_7fd5, 0xbc52_bb36_e6b3_a2af),
    (0x3fd8_daee_6a60_c961, 0x3c6a_4e61_8fb9_2468),
    (0x3fda_4124_b2fe_50cb, 0x3c6f_b5f3_020a_46f5),
    (0x3fdb_acff_156c_79d7, 0xbc66_c536_6444_681a),
    (0x3fdd_1e94_4f6f_bdaa, 0xbc7d_cb87_49a6_4f6e),
    (0x3fde_95fb_7a7a_88f8, 0xbc7c_c04a_2491_ed0a),
    (0x3fe0_09a6_068f_6a8c, 0xbc8a_29a3_2247_3bb6),
    (0x3fe0_cb4e_ee42_c98b, 0xbc8f_511e_f22f_5ad5),
    (0x3fe1_9004_8ef6_0020, 0xbc81_ed92_5f89_3d67),
    (0x3fe2_57d3_3413_7dff, 0x3c8b_99bf_e7ce_9f9e),
    (0x3fe3_22c7_5a96_3b98, 0x3c73_718f_7053_4e8a),
    (0x3fe3_f0ed_b1d1_8acd, 0xbc87_d3f4_0a14_6f34),
];
/// `1/n!`, `n = 0, …, 33`.
pub(super) const INV_FACT: [(u64, u64); 34] = [
    (0x3ff0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3ff0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3fe0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3fc5_5555_5555_5555, 0x3c65_5555_5555_5555),
    (0x3fa5_5555_5555_5555, 0x3c45_5555_5555_5555),
    (0x3f81_1111_1111_1111, 0x3c01_1111_1111_1111),
    (0x3f56_c16c_16c1_6c17, 0xbbef_49f4_9f49_f49f),
    (0x3f2a_01a0_1a01_a01a, 0x3b6a_01a0_1a01_a01a),
    (0x3efa_01a0_1a01_a01a, 0x3b3a_01a0_1a01_a01a),
    (0x3ec7_1de3_a556_c734, 0xbb6c_154f_8ddc_6c00),
    (0x3e92_7e4f_b778_9f5c, 0x3b3c_bbc0_5b4f_a99a),
    (0x3e5a_e645_67f5_44e4, 0xbafc_062e_06d1_f209),
    (0x3e21_eed8_eff8_d898, 0xbac2_aec9_59e1_4c06),
    (0x3de6_1246_13a8_6d09, 0x3a8f_28e0_cc74_8ebe),
    (0x3da9_3974_a8c0_7c9d, 0x3a30_5d6f_8a2e_fd1f),
    (0x3d6a_e7f3_e733_b81f, 0x39e1_d865_6b0e_e8cb),
    (0x3d2a_e7f3_e733_b81f, 0x39a1_d865_6b0e_e8cb),
    (0x3ce9_52c7_7030_ad4a, 0x398a_c981_465d_dc6c),
    (0x3ca6_8278_63b9_7d97, 0x394e_ec01_221a_8b0b),
    (0x3c62_f49b_4681_4157, 0x38f2_650f_61db_dcb4),
    (0x3c1e_542b_a402_0225, 0x387e_a72b_4afe_3c2f),
    (0x3bd7_1b8e_f6dc_f572, 0xb87d_043a_e40c_4647),
    (0x3b90_ce39_6db7_f853, 0xb83a_ebcd_bd20_331c),
    (0x3b47_61b4_1316_381a, 0xb7d3_423c_7d91_404f),
    (0x3aff_2cf0_1972_f578, 0xb789_ada5_fcc1_ab14),
    (0x3ab3_f3cc_dd16_5fa9, 0xb745_8dda_df34_4487),
    (0x3a68_8e85_fc6a_4e5a, 0xb707_1c37_ebd1_6540),
    (0x3a1d_1ab1_c2dc_cea3, 0x36a0_54d0_c78a_ea14),
    (0x39d0_a18a_2635_085d, 0x366b_9e2e_28e1_aa54),
    (0x3982_59f9_8b43_58ad, 0x362e_af8c_39dd_9bc5),
    (0x3933_932c_5047_d60e, 0x35d8_32b7_b530_a627),
    (0x38e4_34d2_e783_f5bc, 0x3580_b87b_91be_9aff),
    (0x3894_34d2_e783_f5bc, 0x3530_b87b_91be_9aff),
    (0x3843_9812_54dd_0d52, 0xb4e2_b1f4_c801_5a2f),
];
/// `1/n`, `n = 1, …, 41` (entry 0 unused).
pub(super) const INV_INT: [(u64, u64); 42] = [
    (0, 0),
    (0x3ff0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3fe0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3fd5_5555_5555_5555, 0x3c75_5555_5555_5555),
    (0x3fd0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3fc9_9999_9999_999a, 0xbc69_9999_9999_999a),
    (0x3fc5_5555_5555_5555, 0x3c65_5555_5555_5555),
    (0x3fc2_4924_9249_2492, 0x3c62_4924_9249_2492),
    (0x3fc0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3fbc_71c7_1c71_c71c, 0x3c5c_71c7_1c71_c71c),
    (0x3fb9_9999_9999_999a, 0xbc59_9999_9999_999a),
    (0x3fb7_45d1_745d_1746, 0xbc47_45d1_745d_1746),
    (0x3fb5_5555_5555_5555, 0x3c55_5555_5555_5555),
    (0x3fb3_b13b_13b1_3b14, 0xbc53_b13b_13b1_3b14),
    (0x3fb2_4924_9249_2492, 0x3c52_4924_9249_2492),
    (0x3fb1_1111_1111_1111, 0x3c31_1111_1111_1111),
    (0x3fb0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3fae_1e1e_1e1e_1e1e, 0x3c2e_1e1e_1e1e_1e1e),
    (0x3fac_71c7_1c71_c71c, 0x3c4c_71c7_1c71_c71c),
    (0x3faa_f286_bca1_af28, 0x3c4a_f286_bca1_af28),
    (0x3fa9_9999_9999_999a, 0xbc49_9999_9999_999a),
    (0x3fa8_6186_1861_8618, 0x3c48_6186_1861_8618),
    (0x3fa7_45d1_745d_1746, 0xbc37_45d1_745d_1746),
    (0x3fa6_42c8_590b_2164, 0x3c36_42c8_590b_2164),
    (0x3fa5_5555_5555_5555, 0x3c45_5555_5555_5555),
    (0x3fa4_7ae1_47ae_147b, 0xbc2e_b851_eb85_1eb8),
    (0x3fa3_b13b_13b1_3b14, 0xbc43_b13b_13b1_3b14),
    (0x3fa2_f684_bda1_2f68, 0x3c42_f684_bda1_2f68),
    (0x3fa2_4924_9249_2492, 0x3c42_4924_9249_2492),
    (0x3fa1_a7b9_611a_7b96, 0x3c21_a7b9_611a_7b96),
    (0x3fa1_1111_1111_1111, 0x3c21_1111_1111_1111),
    (0x3fa0_8421_0842_1084, 0x3c30_8421_0842_1084),
    (0x3fa0_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3f9f_07c1_f07c_1f08, 0xbc2f_07c1_f07c_1f08),
    (0x3f9e_1e1e_1e1e_1e1e, 0x3c1e_1e1e_1e1e_1e1e),
    (0x3f9d_41d4_1d41_d41d, 0x3c30_7507_5075_0750),
    (0x3f9c_71c7_1c71_c71c, 0x3c3c_71c7_1c71_c71c),
    (0x3f9b_acf9_14c1_bad0, 0xbc3b_acf9_14c1_bad0),
    (0x3f9a_f286_bca1_af28, 0x3c3a_f286_bca1_af28),
    (0x3f9a_41a4_1a41_a41a, 0x3c30_6906_9069_0690),
    (0x3f99_9999_9999_999a, 0xbc39_9999_9999_999a),
    (0x3f98_f9c1_8f9c_18fa, 0xbc2f_3831_f383_1f38),
];
/// `ln(1 + j/64)`, `j = −18, …, 26` (index `j + 18`; Algorithm L).
pub(super) const LN_C: [(u64, u64); 45] = [
    (0xbfd5_22ae_0738_a3d8, 0x3c68_f7e9_b38a_6979),
    (0xbfd3_c252_7733_3184, 0x3c72_ad27_e50a_8ec6),
    (0xbfd2_6962_1134_db92, 0xbc7e_0efa_dd9d_b02b),
    (0xbfd1_178e_8227_e47c, 0x3c60_e63a_5f01_c691),
    (0xbfcf_991c_6cb3_b379, 0xbc6f_6650_66f9_80a2),
    (0xbfcd_1037_f265_5e7b, 0xbc66_0629_2424_71a2),
    (0xbfca_93ed_3c8a_d9e3, 0xbc6b_cafa_9de9_7203),
    (0xbfc8_23c1_6551_a3c2, 0x3c61_232c_e70b_e781),
    (0xbfc5_bf40_6b54_3db2, 0x3c21_f5b4_4c0d_f7e7),
    (0xbfc3_65fc_b015_9016, 0xbc57_d411_a5b9_44ad),
    (0xbfc1_178e_8227_e47c, 0x3c50_e63a_5f01_c691),
    (0xbfbd_a727_6384_46a2, 0xbc54_01fa_7173_3019),
    (0xbfb9_335e_5d59_4989, 0x3c54_78a8_5704_ccb7),
    (0xbfb4_d311_5d20_7eac, 0xbc57_69f4_2c78_42cc),
    (0xbfb0_8598_b59e_3a07, 0x3c5d_d700_9902_bf32),
    (0xbfa8_94aa_149f_b343, 0xbc3a_8be9_7660_a23d),
    (0xbfa0_415d_89e7_4444, 0xbc4c_05cf_1d75_3622),
    (0xbf90_2056_5893_5847, 0xbc32_7c8e_8416_e71f),
    (0x0000_0000_0000_0000, 0x0000_0000_0000_0000),
    (0x3f8f_c0a8_b0fc_03e4, 0xbc18_3092_c596_42a1),
    (0x3f9f_829b_0e78_3300, 0x3c33_3e3f_04f1_ef23),
    (0x3fa7_7458_f632_dcfc, 0x3c41_8d3c_a87b_9296),
    (0x3faf_0a30_c011_62a6, 0x3c48_5f32_5c5b_bacd),
    (0x3fb3_41d7_961b_d1d1, 0xbc5b_599f_227b_ecbb),
    (0x3fb6_f0d2_8ae5_6b4c, 0xbc59_06d9_9184_b992),
    (0x3fba_926d_3a4a_d563, 0x3c59_42f4_8aa7_0ea9),
    (0x3fbe_2707_6e2a_f2e6, 0xbc36_1578_001e_0162),
    (0x3fc0_d77e_7cd0_8e59, 0x3c69_a5dc_5e90_30ac),
    (0x3fc2_9552_f81f_f523, 0x3c63_0177_1c40_7dbf),
    (0x3fc4_4d2b_6ccb_7d1e, 0x3c69_f4f6_543e_1f88),
    (0x3fc5_ff30_70a7_93d4, 0xbc5b_c60e_fafc_6f6e),
    (0x3fc7_ab89_0210_d909, 0x3c4b_e36b_2d6a_0608),
    (0x3fc9_525a_9cf4_56b4, 0x3c6d_904c_1d4e_2e26),
    (0x3fca_f3c9_4e80_bff3, 0xbc53_98cf_f364_1985),
    (0x3fcc_8ff7_c79a_9a22, 0xbc64_f689_f843_4012),
    (0x3fce_2707_6e2a_f2e6, 0xbc46_1578_001e_0162),
    (0x3fcf_b918_6d5e_3e2b, 0xbc6c_aaae_64f2_1acb),
    (0x3fd0_a324_e273_90e3, 0x3c77_dcfd_e806_1c03),
    (0x3fd1_675c_abab_a60e, 0x3c2c_e63e_ab88_3717),
    (0x3fd2_2941_fbcf_7966, 0xbc77_6f5e_b096_28af),
    (0x3fd2_e8e2_bae1_1d31, 0xbc78_f4cd_b95e_bdf9),
    (0x3fd3_a64c_5569_45ea, 0xbc6c_6865_1945_f97c),
    (0x3fd4_618b_c21c_5ec2, 0x3c7f_42de_cdec_cf1d),
    (0x3fd5_1aad_872d_f82d, 0x3c43_927a_c19f_55e3),
    (0x3fd5_d1bd_bf58_09ca, 0x3c74_2363_83dc_7fe1),
];
/// `ln(2(1 + j/64))`, `j = −18, …, 0` (index `j + 18`; Algorithm L, `e = 1`, `y < 1`).
pub(super) const LN_2C: [(u64, u64); 19] = [
    (0x3fd7_39d7_f6bb_d007, 0xbc78_c76c_eb01_4b04),
    (0x3fd8_9a33_86c1_425b, 0xbc72_9639_dfbb_f0fb),
    (0x3fd9_f323_ecbf_984c, 0xbc4a_92e5_1321_7f5c),
    (0x3fdb_44f7_7bcc_8f63, 0xbc7c_d044_9545_9c78),
    (0x3fdc_8ff7_c79a_9a22, 0xbc74_f689_f843_4012),
    (0x3fdd_d46a_04c1_c4a1, 0xbc70_4676_56d8_b892),
    (0x3fdf_128f_5faf_06ed, 0xbc73_28df_13bb_38c3),
    (0x3fe0_2552_a5a5_d0ff, 0xbc7c_b1cb_5140_8c00),
    (0x3fe0_be72_e425_2a83, 0xbc82_59da_1133_0801),
    (0x3fe1_54c3_d2f4_d5ea, 0xbc85_9c33_171a_6876),
    (0x3fe1_e85f_5e70_40d0, 0x3c7e_f62c_d2f9_f1e3),
    (0x3fe2_795e_1289_b11b, 0xbc64_87c0_c246_978e),
    (0x3fe3_07d7_334f_10be, 0x3c6f_b590_a1f5_66da),
    (0x3fe3_93e0_d356_2a1a, 0xbc85_8eef_67f2_483a),
    (0x3fe4_1d8f_e846_72ae, 0x3c89_192f_30bd_1806),
    (0x3fe4_a4f8_5db0_3ebb, 0x3c31_3dfa_3d37_61b6),
    (0x3fe5_2a2d_265b_c5ab, 0xbc61_8837_50ea_4d0a),
    (0x3fe5_ad40_4c35_9f2d, 0xbc43_5955_683f_7196),
    (0x3fe6_2e42_fefa_39ef, 0x3c7a_bc9e_3b39_803f),
];
/// `ln((1 + j/64)/2)`, `j = 0, …, 26` (Algorithm L, `e = −1`, `y ≥ 1`).
pub(super) const LN_C_2: [(u64, u64); 27] = [
    (0xbfe6_2e42_fefa_39ef, 0xbc7a_bc9e_3b39_803f),
    (0xbfe5_af40_5c36_49e0, 0x3c86_714f_bcd8_135b),
    (0xbfe5_322e_2686_7857, 0xbc79_88ba_4aea_614d),
    (0xbfe4_b6fd_6f97_0c1f, 0xbc8c_457b_5315_06f6),
    (0xbfe4_3d9f_f2f9_23c5, 0x3c78_4f48_1051_f71a),
    (0xbfe3_c608_0c36_bfb5, 0xbc71_9306_03d8_7b6e),
    (0xbfe3_5028_ad9d_8c86, 0x3c7f_01ab_6065_515c),
    (0xbfe2_dbf5_57b0_df43, 0x3c7b_941e_e770_436b),
    (0xbfe2_6962_1134_db92, 0xbc8e_0efa_dd9d_b02b),
    (0xbfe1_f863_5fc6_1659, 0x3c72_164f_f40e_9817),
    (0xbfe1_88ee_40f2_3ca6, 0xbc88_9df1_568c_a0b0),
    (0xbfe1_1af8_23c7_5aa8, 0x3c89_1eee_7772_c7c2),
    (0xbfe0_ae76_e2d0_54fa, 0xbc80_d710_fcfc_4e0d),
    (0xbfe0_4360_be76_03ad, 0x3c61_7f9e_54e7_8104),
    (0xbfdf_b358_af7a_4884, 0xbc67_e8f0_5924_d259),
    (0xbfde_e2a1_56b4_13e5, 0x3c17_4b71_fb5e_57e3),
    (0xbfde_148a_1a27_26ce, 0x3c7a_c81c_c8a4_dfb8),
    (0xbfdd_4902_46de_fa6b, 0xbc7d_7f4d_3b3d_406b),
    (0xbfdc_7ff9_c745_54c9, 0xbc62_23ea_db65_1b4a),
    (0xbfdb_b961_1b80_e2fb, 0xbc46_fd02_999b_21e1),
    (0xbfda_f529_5248_cdd0, 0xbc79_d56c_45dd_3e86),
    (0xbfda_3344_0224_fa79, 0x3c6b_a806_2860_ae23),
    (0xbfd9_73a3_4313_56ae, 0x3c68_9d28_16cf_838f),
    (0xbfd8_b639_a88b_2df5, 0x3c77_0f2f_3823_8303),
    (0xbfd7_fafa_3bd8_151c, 0x3c52_1902_4acd_3b77),
    (0xbfd7_41d8_76c6_7bb1, 0xbc78_4a4e_e305_9583),
    (0xbfd6_8ac8_3e9c_6a14, 0xbc5a_64ea_dd74_0178),
];
/// `atan(c)` for `c = 2^e (1 + (2i + 1)/32)`, `e = −6, …, 2`, `i = 0, …, 15` (index `16(e + 6) + i`; Algorithm T).
pub(super) const ATAN_C: [(u64, u64); 144] = [
    (0x3f90_7fa2_6dbb_46db, 0xbc3d_69b7_cc28_6f51), // atan(0.01611328125)
    (0x3f91_7f90_5dac_abec, 0xbc3a_d1e8_91a1_4cf4), // atan(0.01708984375)
    (0x3f92_7f7c_1df1_e80c, 0x3c2b_74a3_3a1b_2e9a), // atan(0.01806640625)
    (0x3f93_7f65_8e9a_2b38, 0x3c3d_89d6_6c47_fca2), // atan(0.01904296875)
    (0x3f94_7f4c_8fb6_60b2, 0x3c3e_6227_0f7c_2d07), // atan(0.02001953125)
    (0x3f95_7f31_0159_46e3, 0xbc26_6101_c3b5_ddd9), // atan(0.02099609375)
    (0x3f96_7f12_c397_8735, 0x3c37_d371_26c8_ab1e), // atan(0.02197265625)
    (0x3f97_7ef1_b687_cdf3, 0xbc3d_2f41_3c7e_b9e0), // atan(0.02294921875)
    (0x3f98_7ecd_ba42_e215, 0xbc22_d373_6270_08af), // atan(0.02392578125)
    (0x3f99_7ea6_aee3_bd1a, 0x3c3e_6e29_4c2a_d53d), // atan(0.02490234375)
    (0x3f9a_7e7c_7487_a2d3, 0xbc31_e641_e313_f225), // atan(0.02587890625)
    (0x3f9b_7e4e_eb4e_3927, 0x3c2b_4ceb_31f0_ccb6), // atan(0.02685546875)
    (0x3f9c_7e1d_f359_9fe1, 0xbc37_f466_72e8_7c88), // atan(0.02783203125)
    (0x3f9d_7de9_6cce_8867, 0xbc2c_f6a8_4a76_69f0), // atan(0.02880859375)
    (0x3f9e_7db1_37d4_4d7c, 0xbc15_7f24_4407_0467), // atan(0.02978515625)
    (0x3f9f_7d75_3495_0af3, 0x3c13_fc3d_93c9_47a1), // atan(0.03076171875)
    (0x3fa0_7e89_e3ab_ee7e, 0xbc14_87ba_8ef8_f523), // atan(0.0322265625)
    (0x3fa1_7e41_b2bd_eb61, 0xbc2e_c808_e694_1860), // atan(0.0341796875)
    (0x3fa2_7df0_c70b_94df, 0x3c3e_dc1f_c47f_3298), // atan(0.0361328125)
    (0x3fa3_7d96_a187_5a50, 0x3c41_4630_cae3_54c7), // atan(0.0380859375)
    (0x3fa4_7d32_c33f_3cb4, 0x3c4a_00db_0726_717d), // atan(0.0400390625)
    (0x3fa5_7cc4_ad5e_46d1, 0x3c4a_f5b6_92e5_208c), // atan(0.0419921875)
    (0x3fa6_7c4b_e12e_0476, 0x3c2e_dbef_c278_9435), // atan(0.0439453125)
    (0x3fa7_7bc7_e017_f8db, 0xbc31_b274_6d8f_a6a3), // atan(0.0458984375)
    (0x3fa8_7b38_2ba7_1414, 0x3c34_38cb_47ba_dbd9), // atan(0.0478515625)
    (0x3fa9_7a9c_4589_278d, 0xbc43_a5d9_aced_edc3), // atan(0.0498046875)
    (0x3faa_79f3_af90_597c, 0x3c2f_c19b_de18_16d2), // atan(0.0517578125)
    (0x3fab_793d_ebb4_9750, 0x3c2a_ad65_4cd7_39d1), // atan(0.0537109375)
    (0x3fac_787a_7c15_06fd, 0x3bf9_93ff_6d7d_0532), // atan(0.0556640625)
    (0x3fad_77a8_e2f9_772c, 0xbc1f_361e_817d_1ba4), // atan(0.0576171875)
    (0x3fae_76c8_a2d3_ce3c, 0xbc4d_d1a3_cdad_c8b8), // atan(0.0595703125)
    (0x3faf_75d9_3e41_7809, 0x3c49_1c53_84f3_8a8d), // atan(0.0615234375)
    (0x3fb0_7a2a_58a0_c16f, 0x3c52_86a0_aa8f_bfd2), // atan(0.064453125)
    (0x3fb1_790a_88ac_a931, 0x3c5c_57fd_0828_1008), // atan(0.068359375)
    (0x3fb2_77c8_0c02_ec4d, 0x3c58_69be_03c4_d7f0), // atan(0.072265625)
    (0x3fb3_7660_f1a6_b5d8, 0x3c50_0c2b_ea11_5ef0), // atan(0.076171875)
    (0x3fb4_74d3_4a4b_bb9d, 0xbc10_d396_5910_af34), // atan(0.080078125)
    (0x3fb5_731d_286c_4ecb, 0xbc4e_6e75_4b5c_9fd0), // atan(0.083984375)
    (0x3fb6_713c_a05f_38b3, 0x3c28_844b_e8e0_089b), // atan(0.087890625)
    (0x3fb7_6f2f_c86d_613d, 0xbc40_517b_6267_cdb9), // atan(0.091796875)
    (0x3fb8_6cf4_b8e7_3cbf, 0xbc5d_cdd9_15cf_736b), // atan(0.095703125)
    (0x3fb9_6a89_8c39_fefb, 0xbc51_cfa6_eef4_07ce), // atan(0.099609375)
    (0x3fba_67ec_5f04_910a, 0x3c59_eda5_1bd1_2082), // atan(0.103515625)
    (0x3fbb_651b_502c_480a, 0xbc5c_46fc_8733_1ba0), // atan(0.107421875)
    (0x3fbc_6214_80f1_5a6a, 0xbc3c_fcca_a3f6_6870), // atan(0.111328125)
    (0x3fbd_5ed6_1503_11dc, 0x3c4e_b3fd_6855_286c), // atan(0.115234375)
    (0x3fbe_5b5e_3293_b7cf, 0x3c4d_4aae_80ff_2fd5), // atan(0.119140625)
    (0x3fbf_57ab_026c_3a90, 0xbc4c_26c3_afc8_b17a), // atan(0.123046875)
    (0x3fc0_68d5_8421_2b3e, 0xbc69_e2d2_8301_9bfd), // atan(0.12890625)
    (0x3fc1_6465_4106_0850, 0x3c66_bcee_8ae7_ea92), // atan(0.13671875)
    (0x3fc2_5f6e_171a_535c, 0x3c67_c6d7_bde1_a310), // atan(0.14453125)
    (0x3fc3_59e8_edeb_99a4, 0xbc6a_5fd7_4e46_04c6), // atan(0.15234375)
    (0x3fc4_53ce_c609_2a9e, 0x3c61_f653_b3a5_a78b), // atan(0.16015625)
    (0x3fc5_4d18_ba11_570a, 0x3c61_8282_f288_4073), // atan(0.16796875)
    (0x3fc6_45bf_ffb3_aa74, 0xbc3f_536b_677c_2cb4), // atan(0.17578125)
    (0x3fc7_3dbd_e8a7_d202, 0xbc55_ad0f_6d4a_665d), // atan(0.18359375)
    (0x3fc8_350b_e398_ebc8, 0xbc55_a913_32b9_c90d), // atan(0.19140625)
    (0x3fc9_2ba3_7d05_0272, 0xbc60_d3de_d0ff_4764), // atan(0.19921875)
    (0x3fca_217e_6010_81a6, 0xbc60_def8_a60a_f374), // atan(0.20703125)
    (0x3fcb_1696_574d_780c, 0xbc58_5ab8_fc15_a673), // atan(0.21484375)
    (0x3fcc_0ae5_4d76_8467, 0xbc60_4cdb_f55f_26dc), // atan(0.22265625)
    (0x3fcc_fe65_4e1d_5395, 0x3c64_7b9a_3f71_eafb), // atan(0.23046875)
    (0x3fcd_f110_864c_9d9e, 0xbc35_818b_53bf_4781), // atan(0.23828125)
    (0x3fce_e2e1_451d_980d, 0xbc59_a770_8c46_ba91), // atan(0.24609375)
    (0x3fd0_25fa_5106_65b6, 0xbc76_72df_6832_fa48), // atan(0.2578125)
    (0x3fd1_151a_3624_31ca, 0xbc74_dc8d_c907_7b9f), // atan(0.2734375)
    (0x3fd2_0255_67e4_7c96, 0xbc61_8323_28f4_290e), // atan(0.2890625)
    (0x3fd2_ed98_7a82_3cfe, 0x3c6b_9125_8ea0_12ca), // atan(0.3046875)
    (0x3fd3_d6d1_2927_1134, 0x3c71_37ca_41cc_958a), // atan(0.3203125)
    (0x3fd4_bdee_5868_90e7, 0xbc6e_4dc7_7c22_a757), // atan(0.3359375)
    (0x3fd5_a2e0_175e_0f4e, 0x3c71_3b7a_8f82_e457), // atan(0.3515625)
    (0x3fd6_8597_9f5f_a6fe, 0xbc42_5781_4d1a_da9c), // atan(0.3671875)
    (0x3fd7_6607_5281_7502, 0xbc4d_d117_91cc_7600), // atan(0.3828125)
    (0x3fd8_4422_b8df_95d7, 0x3c7d_76a0_299b_41b6), // atan(0.3984375)
    (0x3fd9_1fde_7cd0_c662, 0x3c71_0741_8805_4b53), // atan(0.4140625)
    (0x3fd9_f930_6616_8002, 0xbc7c_8270_47c9_439a), // atan(0.4296875)
    (0x3fda_d00f_5422_058b, 0x3c7f_c4c3_3891_d2e8), // atan(0.4453125)
    (0x3fdb_a473_3786_24a5, 0x3c75_19a1_b46e_4aff), // atan(0.4609375)
    (0x3fdc_7655_0aad_71f9, 0xbc77_4b8b_ff70_43e4), // atan(0.4765625)
    (0x3fdd_45ae_c9ec_862b, 0x3c68_9421_163e_f92d), // atan(0.4921875)
    (0x3fde_77eb_7f17_5a34, 0x3c70_e53d_c1bf_3435), // atan(0.515625)
    (0x3fe0_039c_73c1_a40c, 0xbc8b_32c9_49c9_d593), // atan(0.546875)
    (0x3fe0_c614_5b5b_43da, 0x3c59_74fa_13b5_404f), // atan(0.578125)
    (0x3fe1_835a_88be_7c13, 0x3c8c_621c_ec00_c301), // atan(0.609375)
    (0x3fe2_3b71_e2cc_9e6a, 0x3c6c_421c_9f38_224e), // atan(0.640625)
    (0x3fe2_ee62_8406_cbca, 0x3c8c_5d5e_9ff0_cf8d), // atan(0.671875)
    (0x3fe3_9c39_1cd4_171a, 0xbc82_3043_31d8_bf46), // atan(0.703125)
    (0x3fe4_4506_5b79_5b56, 0xbc7f_76d0_163f_79c8), // atan(0.734375)
    (0x3fe4_e8de_5bb6_ec04, 0x3c84_a33d_beb3_796c), // atan(0.765625)
    (0x3fe5_87d8_1f73_2fbb, 0xbc75_e5c9_d8c5_a950), // atan(0.796875)
    (0x3fe6_220d_115d_7b8e, 0xbc62_b785_350e_e8c1), // atan(0.828125)
    (0x3fe6_b798_920b_3d99, 0xbc8a_8038_6188_c50e), // atan(0.859375)
    (0x3fe7_4897_8fba_8e0f, 0x3c47_b2a6_1658_84a1), // atan(0.890625)
    (0x3fe7_d528_289f_a093, 0x3c85_6082_1e2f_3aa9), // atan(0.921875)
    (0x3fe8_5d69_576c_c2c5, 0x3c66_b66e_7fc8_b8c3), // atan(0.953125)
    (0x3fe8_e17a_a99c_c05e, 0xbc7e_c182_ab04_2f61), // atan(0.984375)
    (0x3fe9_a000_a935_bd8e, 0x3c75_9411_df0d_ccef), // atan(1.03125)
    (0x3fea_908a_fa5b_1d4a, 0xbc75_d7be_5d5f_808b), // atan(1.09375)
    (0x3feb_7291_b4e2_5bda, 0xbc7c_49cc_26e6_3660), // atan(1.15625)
    (0x3fec_470a_bf2d_3d01, 0x3c76_a61d_bf19_9479), // atan(1.21875)
    (0x3fed_0ee2_2538_86a6, 0x3c82_c9f7_3793_dded), // atan(1.28125)
    (0x3fed_caf8_2dc1_a6f4, 0xbc8f_99cb_3ddd_4790), // atan(1.34375)
    (0x3fee_7c20_4235_0f87, 0xbc60_e14d_8d5a_7dd8), // atan(1.40625)
    (0x3fef_2320_73ae_b172, 0xbc85_f5b3_a2cd_fc2c), // atan(1.46875)
    (0x3fef_c0b1_71ec_926c, 0xbc53_3373_69af_334f), // atan(1.53125)
    (0x3ff0_2abf_692f_6d0c, 0xbc97_e03a_2935_1e05), // atan(1.59375)
    (0x3ff0_7113_c6a9_3a21, 0x3c7c_2bc4_d3a3_e69f), // atan(1.65625)
    (0x3ff0_b39f_4eca_23ae, 0x3c92_5934_545c_016c), // atan(1.71875)
    (0x3ff0_f2a5_d9ff_f026, 0x3c8e_6ac2_e916_1719), // atan(1.78125)
    (0x3ff1_2e65_fa32_aaed, 0xbc9f_25b0_8b14_d8d6), // atan(1.84375)
    (0x3ff1_6719_5a20_3265, 0x3c91_a5ac_a105_c6ae), // atan(1.90625)
    (0x3ff1_9cf5_1b06_03dd, 0xbc84_b79c_f12e_503d), // atan(1.96875)
    (0x3ff1_e8d4_73c5_d5ca, 0x3c94_0b5b_2505_c143), // atan(2.0625)
    (0x3ff2_45b4_faf2_3111, 0xbc9b_cadb_a0fe_318b), // atan(2.1875)
    (0x3ff2_9a33_f97b_dbea, 0x3c92_0768_f82d_028d), // atan(2.3125)
    (0x3ff2_e757_2883_3a54, 0x3c71_6e3e_f732_6bda), // atan(2.4375)
    (0x3ff3_2dfe_01c1_1c21, 0x3c9c_b1af_39d7_5eb5), // atan(2.5625)
    (0x3ff3_6ee7_f2a2_4644, 0xbc92_c820_9756_21fb), // atan(2.6875)
    (0x3ff3_aab9_8641_f26b, 0xbc8d_c349_cc17_5bc7), // atan(2.8125)
    (0x3ff3_e200_aea0_0d99, 0xbc94_794d_da3d_c8fb), // atan(2.9375)
    (0x3ff4_1538_521b_2f98, 0x3c6b_0a24_edb2_ee98), // atan(3.0625)
    (0x3ff4_44cb_3d7d_780c, 0x3c98_4edb_dae1_963f), // atan(3.1875)
    (0x3ff4_7116_95fe_dde2, 0xbc83_69e2_2089_162c), // atan(3.3125)
    (0x3ff4_9a6b_e20c_3a52, 0x3c96_1f86_cbda_e1ab), // atan(3.4375)
    (0x3ff4_c112_bb9f_7c63, 0x3c6f_7466_5000_6c33), // atan(3.5625)
    (0x3ff4_e54a_3b8e_6cf8, 0xbc85_4667_3bfb_75f0), // atan(3.6875)
    (0x3ff5_074a_2a61_2ac3, 0xbc87_0b6f_0046_b390), // atan(3.8125)
    (0x3ff5_2744_00ee_a72b, 0xbc9c_8ca2_6484_4338), // atan(3.9375)
    (0x3ff5_53ce_48a0_4765, 0xbc8f_b2a1_5b01_af76), // atan(4.125)
    (0x3ff5_8990_974d_fc9b, 0xbc93_8724_877f_df56), // atan(4.375)
    (0x3ff5_b9c9_494c_0d73, 0xbc7d_bb3c_b11f_72ae), // atan(4.625)
    (0x3ff5_e545_b9b1_a4c8, 0xbc9b_1f8a_fb3d_d31d), // atan(4.875)
    (0x3ff6_0cad_f03e_444d, 0xbc91_cafc_7209_e76b), // atan(5.125)
    (0x3ff6_308c_a2a1_ee29, 0x3c86_9afb_aa88_c2dc), // atan(5.375)
    (0x3ff6_5155_42ad_f35b, 0x3c91_abca_6117_c655), // atan(5.625)
    (0x3ff6_6f68_9fe6_ecd7, 0x3c72_96b3_ad3a_b6da), // atan(5.875)
    (0x3ff6_8b18_7b9d_2c61, 0xbc9e_6541_4ed7_6ae6), // atan(6.125)
    (0x3ff6_a4aa_53aa_c449, 0xbc84_371a_18cd_c2a7), // atan(6.375)
    (0x3ff6_bc59_952b_f3b1, 0x3c79_77d7_c13d_209c), // atan(6.625)
    (0x3ff6_d259_5b4f_5943, 0x3c96_d3e4_5139_467e), // atan(6.875)
    (0x3ff6_e6d5_d4f4_d24b, 0x3c5b_c3da_dd1e_e93f), // atan(7.125)
    (0x3ff6_f9f5_650f_d3ef, 0x3c99_b070_ed3f_43e4), // atan(7.375)
    (0x3ff7_0bd9_8cd9_6433, 0x3c5a_965f_9810_24a8), // atan(7.625)
    (0x3ff7_1c9f_ab44_14b2, 0x3c9e_d22a_bd9c_91bc), // atan(7.875)
];
/// `⌊2^1408 · 2/π⌋`, big-endian 64-bit words: bit `i` of `2/π` (weight `2^−i`) is bit `1408 − i` of the integer (Algorithm R).
pub(super) const TWO_OVER_PI: [u64; 22] = [
    0xa2f9_836e_4e44_1529,
    0xfc27_57d1_f534_ddc0,
    0xdb62_9599_3c43_9041,
    0xfe51_63ab_debb_c561,
    0xb724_6e3a_424d_d2e0,
    0x0649_2eea_09d1_921c,
    0xfe1d_eb1c_b129_a73e,
    0xe882_35f5_2ebb_4484,
    0xe99c_7026_b45f_7e41,
    0x3991_d639_8353_39f4,
    0x9c84_5f8b_bdf9_283b,
    0x1ff8_97ff_de05_980f,
    0xef2f_118b_5a0a_6d1f,
    0x6d36_7ecf_27cb_09b7,
    0x4f46_3f66_9e5f_ea2d,
    0x7527_bac7_ebe5_f17b,
    0x3d07_39f7_8a52_92ea,
    0x6bfb_5fb1_1f8d_5d08,
    0x5603_3046_fc7b_6bab,
    0xf0cf_bc20_9af4_361d,
    0xa9e3_9161_5ee6_1b08,
    0x6599_855f_14a0_6840,
];
/// `⌊2^320 · π/4⌋`, big-endian 64-bit words (Algorithm R).
pub(super) const PI_OVER_4: [u64; 5] = [
    0xc90f_daa2_2168_c234,
    0xc4c6_628b_80dc_1cd1,
    0x2902_4e08_8a67_cc74,
    0x020b_bea6_3b13_9b22,
    0x514a_0879_8e34_04dd,
];
/// `3π/4` (the special case `atan2(±∞, −∞)`).
pub(super) const FRAC_3PI_4: (u64, u64) = (0x4002_d97c_7f33_21d2, 0x3c9a_7939_4c9e_8a0a);
