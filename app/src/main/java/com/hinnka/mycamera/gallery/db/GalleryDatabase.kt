package com.hinnka.mycamera.gallery.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [GalleryMediaEntity::class],
    version = 1,
    exportSchema = false
)
@androidx.room.TypeConverters(GalleryConverters::class)
abstract class GalleryDatabase : RoomDatabase() {
    abstract fun galleryMediaDao(): GalleryMediaDao

    companion object {
        @Volatile
        private var INSTANCE: GalleryDatabase? = null

        fun getInstance(context: Context): GalleryDatabase {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    GalleryDatabase::class.java,
                    "gallery_media.db"
                )
                    .build()
                    .also { INSTANCE = it }
            }
        }
    }
}
