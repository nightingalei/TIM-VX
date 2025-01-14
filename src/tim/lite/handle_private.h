/****************************************************************************
*
*    Copyright (c) 2021 Vivante Corporation
*
*    Permission is hereby granted, free of charge, to any person obtaining a
*    copy of this software and associated documentation files (the "Software"),
*    to deal in the Software without restriction, including without limitation
*    the rights to use, copy, modify, merge, publish, distribute, sublicense,
*    and/or sell copies of the Software, and to permit persons to whom the
*    Software is furnished to do so, subject to the following conditions:
*
*    The above copyright notice and this permission notice shall be included in
*    all copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
*    DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#ifndef __TIME_LITE_HANDLE_PRIVATE_H__
#define __TIME_LITE_HANDLE_PRIVATE_H__

#include <mutex>
#include "tim/lite/handle.h"
#include "vip_lite.h"

namespace tim {
namespace lite {

class HandleImpl {
    public:
        HandleImpl() : handle_(nullptr) {}
        virtual bool Register(vip_buffer_create_params_t& params) = 0;
        virtual bool Unregister() = 0;
        vip_buffer handle() { return handle_; }
    protected:
        vip_buffer handle_;
};

class UserHandleImpl : public HandleImpl {
    public:
        UserHandleImpl(void* buffer, size_t size)
            : user_buffer_(buffer), user_buffer_size_(size) {}
        bool Register(vip_buffer_create_params_t& params) override;
        bool Unregister() override;
        size_t user_buffer_size() const { return user_buffer_size_; }
        void* user_buffer() { return user_buffer_; }
    private:
        void* user_buffer_;
        size_t user_buffer_size_;
        std::once_flag register_once_;
};

}
}

#endif