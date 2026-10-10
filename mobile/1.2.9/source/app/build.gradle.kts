plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.swamitech.phoenix"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.swamitech.phoenix2"
        minSdk = 26
        targetSdk = 35
        versionCode = 129
        versionName = "1.2.9"
    }


    val releaseStoreFile = System.getenv("PHOENIX_KEYSTORE_FILE")
    val releaseStorePassword = System.getenv("PHOENIX_STORE_PASSWORD")
    val releaseKeyAlias = System.getenv("PHOENIX_KEY_ALIAS")
    val releaseKeyPassword = System.getenv("PHOENIX_KEY_PASSWORD")

    signingConfigs {
        if (!releaseStoreFile.isNullOrBlank() && !releaseStorePassword.isNullOrBlank() &&
            !releaseKeyAlias.isNullOrBlank() && !releaseKeyPassword.isNullOrBlank()) {
            create("production") {
                storeFile = file(releaseStoreFile)
                storePassword = releaseStorePassword
                keyAlias = releaseKeyAlias
                keyPassword = releaseKeyPassword
            }
        }
    }

    buildTypes {
        getByName("release") {
            val hasSigning = !releaseStoreFile.isNullOrBlank() && !releaseStorePassword.isNullOrBlank() &&
                !releaseKeyAlias.isNullOrBlank() && !releaseKeyPassword.isNullOrBlank()
            if (hasSigning) signingConfig = signingConfigs.getByName("production")
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }

    // Keep Java and Kotlin bytecode targets aligned for AGP/Kotlin 2.x.
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures { compose = true }

    externalNativeBuild {
        cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" }
    }

    packaging { resources.excludes += "/META-INF/{AL2.0,LGPL2.1}" }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.compose.ui:ui:1.7.8")
    implementation("androidx.compose.material3:material3:1.3.1")
    implementation("androidx.compose.ui:ui-tooling-preview:1.7.8")
    implementation("androidx.datastore:datastore-preferences:1.1.1")
    implementation("androidx.security:security-crypto:1.1.0-alpha06")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.okhttp3:okhttp-dnsoverhttps:4.12.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
}
