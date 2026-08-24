allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")

    // Keep Kotlin aligned with each plugin's Java target. tflite_flutter 0.12.1
    // compiles for JVM 11, while image_picker_android 0.8.13+19 compiles for
    // JVM 17 under the current Android Gradle plugin.
    val kotlinTarget =
        if (project.name == "image_picker_android") {
            org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
        } else {
            org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11
        }
    tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
        compilerOptions {
            jvmTarget.set(kotlinTarget)
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
