import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kt_edit as K
src = open(sys.argv[1], encoding="utf-8").read()
out = sys.argv[2]
def member(name):
    i = src.index("    private fun %s(" % name)
    # include a preceding KDoc / comment block
    j = i
    while True:
        k = src.rfind("\n", 0, j - 1)
        line = src[k + 1:j - 1] if j > 0 else ""
        if line.strip().startswith(("*", "/**", "//", "*/")) :
            j = k + 1
        else:
            break
    p = src.index("(", i)
    e = K._skip_parens(src, p)
    t = e
    while src[t] in " \t\r\n":
        t += 1
    if src[t] == ":":
        while src[t] != "{" and src[t] != "=":
            t += 1
    if src[t] == "{":
        end = K._skip_braces(src, t)
    else:
        end = src.index("\n", t)       # expression body: one statement (not expected here)
    return src[j:end]
open(os.path.join(out, "VmHarness.kt"), "w", encoding="utf-8").write('''package com.swamitech.phoenix
import android.app.Application
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import com.swamitech.phoenix.net.HfApi
import java.io.File

class VmHarness(private val application: Application) {
    var status: String = ""
    lateinit var hf: HfApi
    @Suppress("UNCHECKED_CAST")
    fun <T : Application> getApplication(): T = application as T

''' + member("resumableToken") + "\n\n" + member("sendVideo") + "\n\n" + member("hasInternet") + "\n}\n")
open(os.path.join(out, "StubApp.kt"), "w").write('''package android.app
open class Application { val contentResolver: android.content.ContentResolver = android.content.ContentResolver()
    fun <T> getSystemService(cls: Class<T>): T? = null }
''')
open(os.path.join(out, "StubPfd.kt"), "w").write('''package android.os
class ParcelFileDescriptor : java.io.Closeable { val statSize: Long = 0L; val fileDescriptor: java.io.FileDescriptor = java.io.FileDescriptor(); override fun close() {} }
''')
open(os.path.join(out, "StubNet.kt"), "w").write('''package android.net
class Network
class NetworkCapabilities { fun hasCapability(c: Int): Boolean = true; companion object { const val NET_CAPABILITY_INTERNET = 12 } }
class ConnectivityManager { val activeNetwork: Network? get() = null; fun getNetworkCapabilities(n: Network): NetworkCapabilities? = null }
''')
print("vm harness written")
