function(find_source_file SRC_FILES arg)
    set(file_list "")
    message(STATUS "[*] searching file: ${arg};${ARGN}")

    list(LENGTH ARGV argv_len)
    set(i 1)
    while (i LESS ${argv_len})
        list(GET ARGV ${i} argv_value)
        foreach (file_path ${SRC_DIR})
            file(GLOB file ${file_path}/${argv_value})
            set(file_list ${file_list} ${file})
        endforeach ()

        math(EXPR i "${i} + 1")
    endwhile ()

    set(${SRC_FILES} ${file_list} PARENT_SCOPE)
endfunction()

set(SRC_DIR
        ${ROOT_DIR}/sample/utils
        ${ROOT_DIR}/3rdparty
        ${ROOT_DIR}/src
        ${ROOT_DIR}/src/utils
        ${ROOT_DIR}/src/utils/shape
        ${ROOT_DIR}/src/module
        ${ROOT_DIR}/src/module/device
        ${ROOT_DIR}/src/module/detect
        ${ROOT_DIR}/src/module/depth
        ${ROOT_DIR}/src/module/enhance
        ${ROOT_DIR}/src/module/relocation
        ${ROOT_DIR}/src/data
        ${ROOT_DIR}/src/instance
        ${ROOT_DIR}/src/map
        ${ROOT_DIR}/src/mot
        )