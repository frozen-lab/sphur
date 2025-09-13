{
  "targets": [
    {
      "target_name": "sphur",
      "sources": [ "binding.c" ],
      "include_dirs": [ "<(module_root_dir)/include" ],
      "libraries": [ "<(module_root_dir)/include/libsphur.a" ]
    }
  ]
}
