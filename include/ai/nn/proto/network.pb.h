// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: network.proto

#ifndef PROTOBUF_network_2eproto__INCLUDED
#define PROTOBUF_network_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3004000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3004000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
namespace ncooper {
namespace ai {
namespace nn {
class Neuron;
class NeuronDefaultTypeInternal;
extern NeuronDefaultTypeInternal _Neuron_default_instance_;
}  // namespace nn
}  // namespace ai
}  // namespace ncooper

namespace ncooper {
namespace ai {
namespace nn {

namespace protobuf_network_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::internal::ParseTableField entries[];
  static const ::google::protobuf::internal::AuxillaryParseTableField aux[];
  static const ::google::protobuf::internal::ParseTable schema[];
  static const ::google::protobuf::uint32 offsets[];
  static const ::google::protobuf::internal::FieldMetadata field_metadata[];
  static const ::google::protobuf::internal::SerializationTable serialization_table[];
  static void InitDefaultsImpl();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_network_2eproto

// ===================================================================

class Neuron : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:ncooper.ai.nn.Neuron) */ {
 public:
  Neuron();
  virtual ~Neuron();

  Neuron(const Neuron& from);

  inline Neuron& operator=(const Neuron& from) {
    CopyFrom(from);
    return *this;
  }
  #if LANG_CXX11
  Neuron(Neuron&& from) noexcept
    : Neuron() {
    *this = ::std::move(from);
  }

  inline Neuron& operator=(Neuron&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }
  #endif
  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }
  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const Neuron& default_instance();

  static inline const Neuron* internal_default_instance() {
    return reinterpret_cast<const Neuron*>(
               &_Neuron_default_instance_);
  }
  static PROTOBUF_CONSTEXPR int const kIndexInFileMessages =
    0;

  void Swap(Neuron* other);
  friend void swap(Neuron& a, Neuron& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Neuron* New() const PROTOBUF_FINAL { return New(NULL); }

  Neuron* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const Neuron& from);
  void MergeFrom(const Neuron& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(Neuron* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated float weights = 2;
  int weights_size() const;
  void clear_weights();
  static const int kWeightsFieldNumber = 2;
  float weights(int index) const;
  void set_weights(int index, float value);
  void add_weights(float value);
  const ::google::protobuf::RepeatedField< float >&
      weights() const;
  ::google::protobuf::RepeatedField< float >*
      mutable_weights();

  // required int32 id = 1;
  bool has_id() const;
  void clear_id();
  static const int kIdFieldNumber = 1;
  ::google::protobuf::int32 id() const;
  void set_id(::google::protobuf::int32 value);

  // optional float bias = 3;
  bool has_bias() const;
  void clear_bias();
  static const int kBiasFieldNumber = 3;
  float bias() const;
  void set_bias(float value);

  // @@protoc_insertion_point(class_scope:ncooper.ai.nn.Neuron)
 private:
  void set_has_id();
  void clear_has_id();
  void set_has_bias();
  void clear_has_bias();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::internal::HasBits<1> _has_bits_;
  mutable int _cached_size_;
  ::google::protobuf::RepeatedField< float > weights_;
  ::google::protobuf::int32 id_;
  float bias_;
  friend struct protobuf_network_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Neuron

// required int32 id = 1;
inline bool Neuron::has_id() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void Neuron::set_has_id() {
  _has_bits_[0] |= 0x00000001u;
}
inline void Neuron::clear_has_id() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void Neuron::clear_id() {
  id_ = 0;
  clear_has_id();
}
inline ::google::protobuf::int32 Neuron::id() const {
  // @@protoc_insertion_point(field_get:ncooper.ai.nn.Neuron.id)
  return id_;
}
inline void Neuron::set_id(::google::protobuf::int32 value) {
  set_has_id();
  id_ = value;
  // @@protoc_insertion_point(field_set:ncooper.ai.nn.Neuron.id)
}

// repeated float weights = 2;
inline int Neuron::weights_size() const {
  return weights_.size();
}
inline void Neuron::clear_weights() {
  weights_.Clear();
}
inline float Neuron::weights(int index) const {
  // @@protoc_insertion_point(field_get:ncooper.ai.nn.Neuron.weights)
  return weights_.Get(index);
}
inline void Neuron::set_weights(int index, float value) {
  weights_.Set(index, value);
  // @@protoc_insertion_point(field_set:ncooper.ai.nn.Neuron.weights)
}
inline void Neuron::add_weights(float value) {
  weights_.Add(value);
  // @@protoc_insertion_point(field_add:ncooper.ai.nn.Neuron.weights)
}
inline const ::google::protobuf::RepeatedField< float >&
Neuron::weights() const {
  // @@protoc_insertion_point(field_list:ncooper.ai.nn.Neuron.weights)
  return weights_;
}
inline ::google::protobuf::RepeatedField< float >*
Neuron::mutable_weights() {
  // @@protoc_insertion_point(field_mutable_list:ncooper.ai.nn.Neuron.weights)
  return &weights_;
}

// optional float bias = 3;
inline bool Neuron::has_bias() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void Neuron::set_has_bias() {
  _has_bits_[0] |= 0x00000002u;
}
inline void Neuron::clear_has_bias() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void Neuron::clear_bias() {
  bias_ = 0;
  clear_has_bias();
}
inline float Neuron::bias() const {
  // @@protoc_insertion_point(field_get:ncooper.ai.nn.Neuron.bias)
  return bias_;
}
inline void Neuron::set_bias(float value) {
  set_has_bias();
  bias_ = value;
  // @@protoc_insertion_point(field_set:ncooper.ai.nn.Neuron.bias)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)


}  // namespace nn
}  // namespace ai
}  // namespace ncooper

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_network_2eproto__INCLUDED
