package com.hinnka.mycamera.gallery.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.hinnka.mycamera.raw.RawToneMappingParameters

@Database(
    entities = [GalleryMediaEntity::class],
    version = 19,
    exportSchema = false
)
@androidx.room.TypeConverters(GalleryConverters::class)
abstract class GalleryDatabase : RoomDatabase() {
    abstract fun galleryMediaDao(): GalleryMediaDao

    companion object {
        @Volatile
        private var INSTANCE: GalleryDatabase? = null

        private val MIGRATION_1_2 = object : androidx.room.migration.Migration(1, 2) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawDROEnabled INTEGER")
            }
        }

        private val MIGRATION_2_3 = object : androidx.room.migration.Migration(2, 3) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN applyEffectsToVideo INTEGER NOT NULL DEFAULT 0")
            }
        }

        private val MIGRATION_3_4 = object : androidx.room.migration.Migration(3, 4) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN spectralFilmEnabled INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN spectralFilmStock TEXT")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN spectralFilmPrint TEXT")
            }
        }

        private val MIGRATION_4_5 = object : androidx.room.migration.Migration(4, 5) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN spectralFilmCDensityGain REAL NOT NULL DEFAULT 1.0")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN spectralFilmMDensityGain REAL NOT NULL DEFAULT 1.0")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN spectralFilmYDensityGain REAL NOT NULL DEFAULT 1.0")
            }
        }

        private val MIGRATION_5_6 = object : androidx.room.migration.Migration(5, 6) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN recipe_bloom REAL")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN baseline_recipe_bloom REAL")
            }
        }

        private val MIGRATION_6_7 = object : androidx.room.migration.Migration(6, 7) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN recipe_softLight REAL")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN baseline_recipe_softLight REAL")
            }
        }

        private val MIGRATION_7_8 = object : androidx.room.migration.Migration(7, 8) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawHighlightsAdjustment REAL")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawShadowsAdjustment REAL")
            }
        }

        private val MIGRATION_8_9 = object : androidx.room.migration.Migration(8, 9) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawColorEngine TEXT NOT NULL DEFAULT 'AdobeCurve'")
            }
        }

        private val MIGRATION_9_10 = object : androidx.room.migration.Migration(9, 10) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawCfaCorrectionMode TEXT DEFAULT 'Default'")
            }
        }

        private val MIGRATION_10_11 = object : androidx.room.migration.Migration(10, 11) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE gallery_media ADD COLUMN rawAgxBlackRelativeExposure REAL NOT NULL DEFAULT " +
                        RawToneMappingParameters.AGX_BLACK_RELATIVE_EXPOSURE_DEFAULT
                )
                db.execSQL(
                    "ALTER TABLE gallery_media ADD COLUMN rawAgxWhiteRelativeExposure REAL NOT NULL DEFAULT " +
                        RawToneMappingParameters.AGX_WHITE_RELATIVE_EXPOSURE_DEFAULT
                )
                db.execSQL(
                    "ALTER TABLE gallery_media ADD COLUMN rawAgxToe REAL NOT NULL DEFAULT " +
                        RawToneMappingParameters.AGX_TOE_DEFAULT
                )
                db.execSQL(
                    "ALTER TABLE gallery_media ADD COLUMN rawAgxShoulder REAL NOT NULL DEFAULT " +
                        RawToneMappingParameters.AGX_SHOULDER_DEFAULT
                )
                db.execSQL(
                    "ALTER TABLE gallery_media ADD COLUMN rawFilmicBlackRelativeExposure REAL NOT NULL DEFAULT " +
                        RawToneMappingParameters.FILMIC_BLACK_RELATIVE_EXPOSURE_DEFAULT
                )
                db.execSQL(
                    "ALTER TABLE gallery_media ADD COLUMN rawFilmicWhiteRelativeExposure REAL NOT NULL DEFAULT " +
                        RawToneMappingParameters.FILMIC_WHITE_RELATIVE_EXPOSURE_DEFAULT
                )
            }
        }

        private val MIGRATION_11_12 = object : androidx.room.migration.Migration(11, 12) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawWhiteLevelMode TEXT DEFAULT 'Default'")
            }
        }

        private val MIGRATION_12_13 = object : androidx.room.migration.Migration(12, 13) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN tonemapMode TEXT")
            }
        }

        private val MIGRATION_13_14 = object : androidx.room.migration.Migration(13, 14) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN captureNoiseReductionLevel INTEGER")
            }
        }

        private val MIGRATION_14_15 = object : androidx.room.migration.Migration(14, 15) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawLensShadingCorrectionEnabled INTEGER")
            }
        }

        private val MIGRATION_15_16 = object : androidx.room.migration.Migration(15, 16) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawGooglePixelToneMap INTEGER NOT NULL DEFAULT 0")
            }
        }

        private val MIGRATION_16_17 = object : androidx.room.migration.Migration(16, 17) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawGooglePixelToneMapExplicit INTEGER NOT NULL DEFAULT 0")
            }
        }

        private val MIGRATION_17_18 = object : androidx.room.migration.Migration(17, 18) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawBlackBorderCropLeftPx INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawBlackBorderCropTopPx INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawBlackBorderCropRightPx INTEGER NOT NULL DEFAULT 0")
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawBlackBorderCropBottomPx INTEGER NOT NULL DEFAULT 0")
            }
        }

        private val MIGRATION_18_19 = object : androidx.room.migration.Migration(18, 19) {
            override fun migrate(db: androidx.sqlite.db.SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE gallery_media ADD COLUMN rawOppoMasterToneMap INTEGER NOT NULL DEFAULT 0")
            }
        }

        fun getInstance(context: Context): GalleryDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    GalleryDatabase::class.java,
                    "gallery_media.db"
                )
                    .addMigrations(
                        MIGRATION_1_2,
                        MIGRATION_2_3,
                        MIGRATION_3_4,
                        MIGRATION_4_5,
                        MIGRATION_5_6,
                        MIGRATION_6_7,
                        MIGRATION_7_8,
                        MIGRATION_8_9,
                        MIGRATION_9_10,
                        MIGRATION_10_11,
                        MIGRATION_11_12,
                        MIGRATION_12_13,
                        MIGRATION_13_14,
                        MIGRATION_14_15,
                        MIGRATION_15_16,
                        MIGRATION_16_17,
                        MIGRATION_17_18,
                        MIGRATION_18_19
                    )
                    .fallbackToDestructiveMigrationOnDowngrade(false)
                    .fallbackToDestructiveMigration(false)
                    .build()
                    .also { INSTANCE = it }
            }
        }
    }
}
