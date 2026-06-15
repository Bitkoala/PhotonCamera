package com.hinnka.mycamera.processor

import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.GLES30
import android.os.Process
import com.hinnka.mycamera.utils.PLog

object GlesGpuScheduler {
    private const val EGL_CONTEXT_MINOR_VERSION_KHR = 0x30FB
    private const val EGL_CONTEXT_PRIORITY_LEVEL_IMG = 0x3100
    private const val EGL_CONTEXT_PRIORITY_LOW_IMG = 0x3103
    private const val EGL_IMG_CONTEXT_PRIORITY = "EGL_IMG_context_priority"

    fun createBackgroundContext(display: EGLDisplay, config: EGLConfig, tag: String): EGLContext {
        if (supportsLowPriorityContext(display)) {
            val lowPriorityContext = EGL14.eglCreateContext(
                display,
                config,
                EGL14.EGL_NO_CONTEXT,
                intArrayOf(
                    EGL14.EGL_CONTEXT_CLIENT_VERSION,
                    3,
                    EGL_CONTEXT_MINOR_VERSION_KHR,
                    1,
                    EGL_CONTEXT_PRIORITY_LEVEL_IMG,
                    EGL_CONTEXT_PRIORITY_LOW_IMG,
                    EGL14.EGL_NONE,
                ),
                0,
            )
            if (lowPriorityContext != EGL14.EGL_NO_CONTEXT) {
                PLog.d(tag, "Created low-priority GLES context for background stacking")
                return lowPriorityContext
            }
            EGL14.eglGetError()
        }

        val context31 = EGL14.eglCreateContext(
            display,
            config,
            EGL14.EGL_NO_CONTEXT,
            intArrayOf(
                EGL14.EGL_CONTEXT_CLIENT_VERSION,
                3,
                EGL_CONTEXT_MINOR_VERSION_KHR,
                1,
                EGL14.EGL_NONE,
            ),
            0,
        )
        if (context31 != EGL14.EGL_NO_CONTEXT) {
            return context31
        }
        EGL14.eglGetError()

        return EGL14.eglCreateContext(
            display,
            config,
            EGL14.EGL_NO_CONTEXT,
            intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE),
            0,
        )
    }

    fun lowerCurrentThreadPriority(tag: String): Int? {
        return try {
            val tid = Process.myTid()
            val originalPriority = Process.getThreadPriority(tid)
            if (originalPriority < Process.THREAD_PRIORITY_BACKGROUND) {
                Process.setThreadPriority(tid, Process.THREAD_PRIORITY_BACKGROUND)
            }
            originalPriority
        } catch (e: RuntimeException) {
            PLog.w(tag, "Failed to lower GLES stacker thread priority", e)
            null
        }
    }

    fun restoreCurrentThreadPriority(originalPriority: Int?, tag: String) {
        if (originalPriority == null) return
        try {
            Process.setThreadPriority(Process.myTid(), originalPriority)
        } catch (e: RuntimeException) {
            PLog.w(tag, "Failed to restore GLES stacker thread priority", e)
        }
    }

    fun yieldToUiRenderer() {
        GLES30.glFlush()
        Thread.yield()
    }

    private fun supportsLowPriorityContext(display: EGLDisplay): Boolean {
        return EGL14.eglQueryString(display, EGL14.EGL_EXTENSIONS)
            ?.split(' ')
            ?.contains(EGL_IMG_CONTEXT_PRIORITY) == true
    }
}
