package com.hinnka.mycamera.gallery.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [GalleryMediaEntity::class],
    version = 9,
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
                        MIGRATION_8_9
                    )
                    .build()
                    .also { INSTANCE = it }
            }
        }
    }
}
